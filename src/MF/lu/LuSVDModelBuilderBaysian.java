package MF.lu;

import annotation.Alpha;
import annotation.Threshold;
import it.unimi.dsi.fastutil.longs.LongCollection;
import mikera.matrixx.Matrix;
import mikera.matrixx.impl.ImmutableMatrix;
import mikera.vectorz.AVector;
import mikera.vectorz.Vector;
import org.grouplens.lenskit.core.Transient;
import org.grouplens.lenskit.data.pref.IndexedPreference;
import org.grouplens.lenskit.data.pref.PreferenceDomain;
import org.grouplens.lenskit.data.snapshot.PreferenceSnapshot;
import org.grouplens.lenskit.iterative.LearningRate;
import org.grouplens.lenskit.iterative.RegularizationTerm;
import org.grouplens.lenskit.iterative.StoppingCondition;
import org.grouplens.lenskit.iterative.TrainingLoopController;
import org.grouplens.lenskit.mf.funksvd.FeatureCount;
import org.grouplens.lenskit.mf.funksvd.FeatureInfo;
import org.grouplens.lenskit.mf.funksvd.InitialFeatureValue;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import javax.inject.Inject;
import javax.inject.Provider;
import java.util.*;

public class LuSVDModelBuilderBaysian implements Provider<LuSVDModel> {

	private final double alpha;
	private int count;
	private double func;

	protected final int featureCount;
	protected final double learningRate;
	protected final double regularization;
	private final PreferenceDomain domain;
	private final StoppingCondition stoppingCondition;
	protected final PreferenceSnapshot snapshot;
	protected final double initialValue;
	private final double threshold;
	protected final Map<Integer, Double> popMap = new HashMap<Integer, Double>();

	@Inject
	public LuSVDModelBuilderBaysian(@Transient @Nonnull PreferenceSnapshot snapshot,
									@FeatureCount int featureCount,
									@InitialFeatureValue double initVal,
									@Threshold double threshold, @Nullable PreferenceDomain dom, @Alpha double alpha, @LearningRate double lrate,
									@RegularizationTerm double reg, StoppingCondition stop) {
		this.featureCount = featureCount;
		this.initialValue = initVal;
		this.snapshot = snapshot;
		this.threshold = threshold;
		domain = dom;
		this.alpha = alpha;
		learningRate = lrate;
		regularization = reg;
		stoppingCondition = stop;
	}


	private void populatePopMap() {
		Collection<IndexedPreference> ratings = snapshot.getRatings();
		double max = 0.0;
		for (IndexedPreference pref : ratings) {
			double val = 0.0;
			if (popMap.containsKey(pref.getItemIndex())) {
				val = popMap.get(pref.getItemIndex());
			}
			val++;
			if (val > max) {
				max = val;
			}
			popMap.put(pref.getItemIndex(), val);
		}
		for (Integer key : popMap.keySet()) {
			double val = popMap.get(key);
			val /= max;
			popMap.put(key, val);
		}
	}

	@Override
	public LuSVDModel get() {
		populatePopMap();

		int userCount = snapshot.getUserIds().size();
		Matrix userFeatures = Matrix.create(userCount, featureCount);

		int itemCount = snapshot.getItemIds().size();
		Matrix itemFeatures = Matrix.create(itemCount, featureCount);

		List<FeatureInfo> featureInfo = new ArrayList<FeatureInfo>(featureCount);

		Vector uvec = Vector.createLength(userCount);
		Vector ivec = Vector.createLength(itemCount);

		for (int f = 0; f < featureCount; f++) {
			uvec.fill(initialValue);
			ivec.fill(initialValue);

			userFeatures.setColumn(f, uvec);
			assert Math.abs(userFeatures.getColumnView(f).elementSum() - uvec.elementSum()) < 1.0e-4 : "user column sum matches";
			itemFeatures.setColumn(f, ivec);
			assert Math.abs(itemFeatures.getColumnView(f).elementSum() - ivec.elementSum()) < 1.0e-4 : "item column sum matches";

			FeatureInfo.Builder fib = new FeatureInfo.Builder(f);
			featureInfo.add(fib.build());
		}

		TrainingLoopController controller = stoppingCondition.newLoop();
		while (controller.keepTraining(0.0)) {
			train(userFeatures, itemFeatures);
		}


		// Wrap the user/item matrices because we won't use or modify them again
		return new LuSVDModel(ImmutableMatrix.wrap(userFeatures),
				ImmutableMatrix.wrap(itemFeatures),
				snapshot.userIndex(), snapshot.itemIndex(),
				featureInfo);
	}

	private void train(Matrix userFeatures, Matrix itemFeatures) {
		count = 0;
		func = 0;
		LongCollection userIds = snapshot.getUserIds();
		for (long usedId : userIds) {
			Collection<IndexedPreference> userRatings = snapshot.getUserRatings(usedId);
			for (IndexedPreference liked : userRatings) {
				if (liked.getValue() <= threshold) {
					continue;
				}
				for (IndexedPreference disliked : userRatings) {
					if (disliked.getValue() > threshold) {
						continue;
					}
					trainPair(userFeatures, itemFeatures, liked, disliked);
				}
			}
		}
		System.out.println("Pairs count: " + count + "; Function value: " + func / count);
	}

	private void trainPair(Matrix userFeatures, Matrix itemFeatures, IndexedPreference liked, IndexedPreference disliked) {
		AVector user = userFeatures.getRow(liked.getUserIndex());
		AVector likedVec = itemFeatures.getRow(liked.getItemIndex());
		AVector dislikedVec = itemFeatures.getRow(disliked.getItemIndex());

		double likedPred = likedVec.dotProduct(user);
		double dislikedPred = dislikedVec.dotProduct(user);
		double rawDiff = likedPred - dislikedPred;

		likedPred = domain.clampValue(likedPred);
		dislikedPred = domain.clampValue(dislikedPred);

		double pop = Math.pow(popMap.get(disliked.getItemIndex()) + 1, alpha);
		double diff = likedPred - dislikedPred;
		func += function(diff, pop);
		if (diff > 0) {
			count++;
		}

		if (rawDiff > domain.getMaximum() - domain.getMinimum()) {
			return;
		}

		if (Double.isNaN(diff) || Double.isInfinite(diff)) {
			System.out.println("diff is " + diff);
		}

		for (int i = 0; i < featureCount; i++) {
			double der = getDerivative(user.get(i), pop, diff);
			double val = likedVec.get(i) + learningRate * (der - regularization * likedVec.get(i));
			if (!Double.isNaN(val) && !Double.isInfinite(val)) {
				likedVec.set(i, val);
			}

			der = getDerivative(-user.get(i), pop, diff);
			val = dislikedVec.get(i) + learningRate * (der - regularization * dislikedVec.get(i));
			if (!Double.isNaN(val) && !Double.isInfinite(val)) {
				dislikedVec.set(i, val);
			}

			der = getDerivative((likedVec.get(i) - dislikedVec.get(i)), pop, diff);
			val = user.get(i) + learningRate * (der - regularization * user.get(i));
			if (!Double.isNaN(val) && !Double.isInfinite(val)) {
				user.set(i, val);
			}
		}
	}

	protected double getDerivative(double a, double pop, double diff) {
		return a * Math.exp(diff) / (1 + Math.exp(diff)) * pop;
	}

	protected double function(double diff, double pop) {
		return Math.log(1 + Math.exp(diff)) * pop;
	}
}
