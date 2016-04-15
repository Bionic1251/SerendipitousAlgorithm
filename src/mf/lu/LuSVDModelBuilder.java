package mf.lu;

import annotation.Alpha;
import annotation.NormMult;
import annotation.R_Threshold;
import annotation.UpdateRule;
import funkSVD.lu.LuFunkSVDUpdateRule;
import funkSVD.lu.UserPreferences;
import it.unimi.dsi.fastutil.longs.LongCollection;
import mikera.matrixx.Matrix;
import mikera.matrixx.impl.ImmutableMatrix;
import mikera.vectorz.AVector;
import mikera.vectorz.Vector;
import org.grouplens.lenskit.core.Transient;
import org.grouplens.lenskit.data.pref.IndexedPreference;
import org.grouplens.lenskit.data.pref.PreferenceDomain;
import org.grouplens.lenskit.data.snapshot.PreferenceSnapshot;
import org.grouplens.lenskit.iterative.StoppingCondition;
import org.grouplens.lenskit.iterative.TrainingLoopController;
import org.grouplens.lenskit.mf.funksvd.FeatureCount;
import org.grouplens.lenskit.mf.funksvd.FeatureInfo;
import org.grouplens.lenskit.mf.funksvd.InitialFeatureValue;
import pop.PopModel;
import util.Settings;
import util.Util;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import javax.inject.Inject;
import javax.inject.Provider;
import java.util.*;

public class LuSVDModelBuilder implements Provider<LuSVDModel> {

	private final double alpha;
	private final double mult;
	private int count;
	private double func;

	protected final int featureCount;
	private final PreferenceDomain domain;
	private final StoppingCondition stoppingCondition;
	protected PreferenceSnapshot snapshot;
	protected final double initialValue;
	private final double threshold;
	private LuFunkSVDUpdateRule rule;
	private UserPreferences userPreferences;
	private PopModel popModel;

	@Inject
	public LuSVDModelBuilder(@Transient @Nonnull PreferenceSnapshot snapshot,
							 @FeatureCount int featureCount,
							 @InitialFeatureValue double initVal,
							 @R_Threshold double threshold, @Alpha double alpha, StoppingCondition stop,
							 @UpdateRule LuFunkSVDUpdateRule rule, @NormMult double mult, PopModel popModel) {
		this.featureCount = featureCount;
		this.initialValue = initVal;
		this.snapshot = snapshot;
		this.threshold = threshold;
		domain = new PreferenceDomain(Settings.MIN, Settings.MAX);
		this.alpha = alpha;
		stoppingCondition = stop;
		this.rule = rule;
		this.mult = mult;
		this.popModel = popModel;
	}

	@Override
	public LuSVDModel get() {
		System.out.println(LuSVDModelBuilder.class);

		userPreferences = new UserPreferences(snapshot, threshold);

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
			itemFeatures.setColumn(f, ivec);

			FeatureInfo.Builder fib = new FeatureInfo.Builder(f);
			featureInfo.add(fib.build());
		}

		TrainingLoopController controller = stoppingCondition.newLoop();
		while (controller.keepTraining(0.0)) {
			train(userFeatures, itemFeatures);
			calculateStatistics(userFeatures, itemFeatures);
		}

		return new LuSVDModel(ImmutableMatrix.wrap(userFeatures),
				ImmutableMatrix.wrap(itemFeatures),
				snapshot.userIndex(), snapshot.itemIndex(),
				featureInfo);
	}

	private void train(Matrix userFeatures, Matrix itemFeatures) {
		LongCollection userIds = snapshot.getUserIds();
		for (long usedId : userIds) {
			List<IndexedPreference> likedItems = userPreferences.getLikedItems(usedId);
			List<IndexedPreference> dislikedItems = userPreferences.getDislikedItems(usedId);
			double normTerm = 1.0 / likedItems.size() / (double) dislikedItems.size();
			for (IndexedPreference liked : likedItems) {
				for (IndexedPreference disliked : dislikedItems) {
					trainPair(userFeatures, itemFeatures, liked, disliked, normTerm);
				}
			}
		}
	}

	private void trainPair(Matrix userFeatures, Matrix itemFeatures, IndexedPreference liked, IndexedPreference disliked, double norm) {
		AVector user = userFeatures.getRow(liked.getUserIndex());
		AVector likedVec = itemFeatures.getRow(liked.getItemIndex());
		AVector dislikedVec = itemFeatures.getRow(disliked.getItemIndex());

		double likedPred = likedVec.dotProduct(user);
		double dislikedPred = dislikedVec.dotProduct(user);

		likedPred = domain.clampValue(likedPred);
		dislikedPred = domain.clampValue(dislikedPred);

		double pop = Math.pow(popModel.getPop(disliked.getItemId()) / popModel.getMax(), alpha);
		pop *= norm * mult;
		double diff = likedPred - dislikedPred;

		for (int i = 0; i < featureCount; i++) {
			likedVec.addAt(i, rule.getNextStep(user.get(i), pop, diff, likedVec.get(i)));

			dislikedVec.addAt(i, rule.getNextStep(-user.get(i), pop, diff, dislikedVec.get(i)));

			//user.addAt(i, rule.getNextStep(likedVec.get(i) - dislikedVec.get(i), pop, diff, user.get(i)));
		}
	}

	private void calculateStatistics(Matrix userFeatures, Matrix itemFeatures) {
		count = 0;
		func = 0;
		LongCollection userIds = snapshot.getUserIds();
		for (long usedId : userIds) {
			List<IndexedPreference> likedItems = userPreferences.getLikedItems(usedId);
			List<IndexedPreference> dislikedItems = userPreferences.getDislikedItems(usedId);
			double normTerm = 1.0 / likedItems.size() / (double) dislikedItems.size();
			for (IndexedPreference liked : likedItems) {
				for (IndexedPreference disliked : dislikedItems) {
					calculateStatisticsByPair(userFeatures, itemFeatures, liked, disliked, normTerm);
				}
			}
		}
		System.out.println("Pairs count: " + count + "; Function value: " + func);
	}

	private void calculateStatisticsByPair(Matrix userFeatures, Matrix itemFeatures, IndexedPreference liked, IndexedPreference disliked, double norm) {
		AVector user = userFeatures.getRow(liked.getUserIndex());
		AVector likedVec = itemFeatures.getRow(liked.getItemIndex());
		AVector dislikedVec = itemFeatures.getRow(disliked.getItemIndex());

		double likedPred = likedVec.dotProduct(user);
		double dislikedPred = dislikedVec.dotProduct(user);

		likedPred = domain.clampValue(likedPred);
		dislikedPred = domain.clampValue(dislikedPred);

		double pop = Math.pow(popModel.getPop(disliked.getItemId()) / popModel.getMax(), alpha);
		pop *= norm * mult;
		double diff = likedPred - dislikedPred;

		func += rule.getFunctionVal(diff, pop);
		if (diff > 0) {
			count++;
		}
	}
}