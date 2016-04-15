package funkSVD.lu;

import annotation.Alpha;
import annotation.R_Threshold;
import annotation.UpdateRule;
import funkSVD.MyTrainingEstimator;
import it.unimi.dsi.fastutil.longs.LongCollection;
import mikera.matrixx.Matrix;
import mikera.matrixx.impl.ImmutableMatrix;
import mikera.vectorz.AVector;
import mikera.vectorz.Vector;
import org.apache.commons.lang3.time.StopWatch;
import org.grouplens.lenskit.core.Transient;
import org.grouplens.lenskit.data.pref.IndexedPreference;
import org.grouplens.lenskit.data.pref.PreferenceDomain;
import org.grouplens.lenskit.data.snapshot.PreferenceSnapshot;
import org.grouplens.lenskit.iterative.StoppingCondition;
import org.grouplens.lenskit.iterative.TrainingLoopController;
import org.grouplens.lenskit.mf.funksvd.FeatureCount;
import org.grouplens.lenskit.mf.funksvd.FeatureInfo;
import org.grouplens.lenskit.mf.funksvd.InitialFeatureValue;
import util.Util;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import javax.inject.Inject;
import javax.inject.Provider;
import java.util.*;

public class LuFunkSVDModelBuilder implements Provider<LuFunkSVDModel> {
	private int count;
	private double func;

	protected final int featureCount;
	protected final PreferenceSnapshot snapshot;
	protected final double initialValue;
	protected Map<Integer, Double> popMap;
	private final double threshold;
	private final PreferenceDomain domain;
	private final double alpha;
	private final StoppingCondition stoppingCondition;
	private UserPreferences userPreferences;

	protected final LuFunkSVDUpdateRule rule;

	@Inject
	public LuFunkSVDModelBuilder(@Transient @Nonnull PreferenceSnapshot snapshot,
								 @UpdateRule LuFunkSVDUpdateRule rule,
								 @FeatureCount int featureCount,
								 @InitialFeatureValue double initVal, @R_Threshold double threshold, @Nullable PreferenceDomain dom,
								 @Alpha double alpha, StoppingCondition stop) {
		this.featureCount = featureCount;
		this.initialValue = initVal;
		this.snapshot = snapshot;
		this.rule = rule;
		this.threshold = threshold;
		domain = dom;
		this.alpha = alpha;
		this.stoppingCondition = stop;
	}

	@Override
	public LuFunkSVDModel get() {
		System.out.println(LuFunkSVDModelBuilder.class);

		userPreferences = new UserPreferences(snapshot, threshold);

		int userCount = snapshot.getUserIds().size();
		Matrix userFeatures = Matrix.create(userCount, featureCount);

		int itemCount = snapshot.getItemIds().size();
		Matrix itemFeatures = Matrix.create(itemCount, featureCount);

		MyTrainingEstimator estimates = rule.makeEstimator(snapshot);

		List<FeatureInfo> featureInfo = new ArrayList<FeatureInfo>(featureCount);

		Vector uvec = Vector.createLength(userCount);
		Vector ivec = Vector.createLength(itemCount);

		for (int f = 0; f < featureCount; f++) {
			StopWatch timer = new StopWatch();
			timer.start();

			uvec.fill(initialValue);
			ivec.fill(initialValue);

			FeatureInfo.Builder fib = new FeatureInfo.Builder(f);
			trainFeature(f, estimates, uvec, ivec, fib);
			summarizeFeature(uvec, ivec, fib);
			featureInfo.add(fib.build());

			estimates.update(uvec, ivec);

			userFeatures.setColumn(f, uvec);
			itemFeatures.setColumn(f, ivec);

			timer.stop();
		}

		return new LuFunkSVDModel(ImmutableMatrix.wrap(userFeatures),
				ImmutableMatrix.wrap(itemFeatures),
				snapshot.userIndex(), snapshot.itemIndex(),
				featureInfo);
	}

	protected void trainFeature(int feature, MyTrainingEstimator estimates,
								Vector userFeatureVector, Vector itemFeatureVector,
								FeatureInfo.Builder fib) {
		System.out.println("Train feature " + feature);
		double trail = initialValue * initialValue * (featureCount - feature - 1);
		TrainingLoopController controller = stoppingCondition.newLoop();
		calculateStatistics(estimates, userFeatureVector, itemFeatureVector, trail);
		while (controller.keepTraining(0.0)) {
			doFeatureIteration(estimates, userFeatureVector, itemFeatureVector, trail);
			calculateStatistics(estimates, userFeatureVector, itemFeatureVector, trail);
		}
	}

	protected void doFeatureIteration(MyTrainingEstimator estimates,
									  Vector userFeatureVector, Vector itemFeatureVector,
									  double trail) {
		LongCollection userIds = snapshot.getUserIds();
		for (long usedId : userIds) {
			List<IndexedPreference> likedItems = userPreferences.getLikedItems(usedId);
			List<IndexedPreference> dislikedItems = userPreferences.getDislikedItems(usedId);
			double normTerm = 1.0 / likedItems.size() / (double) dislikedItems.size();
			for (IndexedPreference liked : likedItems) {
				for (IndexedPreference disliked : dislikedItems) {
					trainPair(userFeatureVector, itemFeatureVector, liked, disliked, estimates, trail, normTerm);
				}
			}
		}
	}

	private void trainPair(Vector userFeatureVector, Vector itemFeatureVector, IndexedPreference liked,
						   IndexedPreference disliked, MyTrainingEstimator estimates, double trail, double norm) {
		double uv = userFeatureVector.get(liked.getUserIndex());
		double likedIV = itemFeatureVector.get(liked.getItemIndex());
		double likedPred = estimates.get(liked) + uv * likedIV + trail;
		double dislikedIV = itemFeatureVector.get(disliked.getItemIndex());
		double dislikedPred = estimates.get(disliked) + uv * dislikedIV + trail;

		dislikedPred = domain.clampValue(dislikedPred);
		likedPred = domain.clampValue(likedPred);

		double pop = Math.pow(popMap.get(disliked.getItemIndex()) + 1, alpha);
		pop *= norm;
		double diff = likedPred - dislikedPred;

		itemFeatureVector.addAt(liked.getItemIndex(), rule.getNextStep(uv, pop, diff, likedIV));
		itemFeatureVector.addAt(disliked.getItemIndex(), rule.getNextStep(-uv, pop, diff, dislikedIV));
		//userFeatureVector.addAt(liked.getUserIndex(), rule.getNextStep(likedIV, pop, diff, uv));
	}

	protected void calculateStatistics(MyTrainingEstimator estimates,
										  Vector userFeatureVector, Vector itemFeatureVector,
										  double trail) {
		count = 0;
		func = 0;
		LongCollection userIds = snapshot.getUserIds();
		for (long usedId : userIds) {
			List<IndexedPreference> likedItems = userPreferences.getLikedItems(usedId);
			List<IndexedPreference> dislikedItems = userPreferences.getDislikedItems(usedId);
			double normTerm = 1.0 / likedItems.size() / (double) dislikedItems.size();
			for (IndexedPreference liked : likedItems) {
				for (IndexedPreference disliked : dislikedItems) {
					calculateStatisticsForPair(userFeatureVector, itemFeatureVector, liked, disliked, estimates, trail, normTerm);
				}
			}
		}
		System.out.println("Pairs count: " + count + "; Function value: " + func);
	}

	private void calculateStatisticsForPair(Vector userFeatureVector, Vector itemFeatureVector, IndexedPreference liked,
											IndexedPreference disliked, MyTrainingEstimator estimates, double trail, double norm) {
		double uv = userFeatureVector.get(liked.getUserIndex());
		double likedIV = itemFeatureVector.get(liked.getItemIndex());
		double likedPred = estimates.get(liked) + uv * likedIV + trail;
		double dislikedIV = itemFeatureVector.get(disliked.getItemIndex());
		double dislikedPred = estimates.get(disliked) + uv * dislikedIV + trail;

		dislikedPred = domain.clampValue(dislikedPred);
		likedPred = domain.clampValue(likedPred);
		double pop = Math.pow(popMap.get(disliked.getItemIndex()) + 1, alpha);
		pop *= norm;
		double diff = likedPred - dislikedPred;
		func += rule.getFunctionVal(diff, pop);
		if (diff > 0) {
			count++;
		}
	}

	protected void summarizeFeature(AVector ufv, AVector ifv, FeatureInfo.Builder fib) {
		fib.setUserAverage(ufv.elementSum() / ufv.length())
				.setItemAverage(ifv.elementSum() / ifv.length())
				.setSingularValue(ufv.magnitude() * ifv.magnitude());
	}
}
