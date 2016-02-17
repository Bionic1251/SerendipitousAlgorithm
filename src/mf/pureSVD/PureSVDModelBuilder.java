package mf.pureSVD;

import it.unimi.dsi.fastutil.longs.LongCollection;
import mikera.matrixx.Matrix;
import mikera.matrixx.impl.ImmutableMatrix;
import mikera.vectorz.AVector;
import mikera.vectorz.Vector;
import org.grouplens.lenskit.core.Transient;
import org.grouplens.lenskit.data.pref.IndexedPreference;
import org.grouplens.lenskit.data.snapshot.PreferenceSnapshot;
import org.grouplens.lenskit.iterative.LearningRate;
import org.grouplens.lenskit.iterative.RegularizationTerm;
import org.grouplens.lenskit.iterative.StoppingCondition;
import org.grouplens.lenskit.iterative.TrainingLoopController;
import org.grouplens.lenskit.mf.funksvd.FeatureCount;
import org.grouplens.lenskit.mf.funksvd.FeatureInfo;
import org.grouplens.lenskit.mf.funksvd.InitialFeatureValue;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.inject.Inject;
import javax.inject.Provider;
import java.util.*;

public class PureSVDModelBuilder implements Provider<PureSVDModel> {
	private static Logger logger = LoggerFactory.getLogger(PureSVDModelBuilder.class);

	protected final int featureCount;
	protected final double learningRate;
	protected final double regularization;
	private final StoppingCondition stoppingCondition;
	protected final PreferenceSnapshot snapshot;
	protected final double initialValue;

	@Inject
	public PureSVDModelBuilder(@Transient @Nonnull PreferenceSnapshot snapshot,
							   @FeatureCount int featureCount,
							   @InitialFeatureValue double initVal, @LearningRate double lrate,
							   @RegularizationTerm double reg, StoppingCondition stop) {
		this.featureCount = featureCount;
		this.initialValue = initVal;
		this.snapshot = snapshot;
		learningRate = lrate;
		regularization = reg;
		stoppingCondition = stop;
	}

	@Override
	public PureSVDModel get() {
		System.out.println(PureSVDModelBuilder.class);
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
		calculateStatistics(userFeatures, itemFeatures);
		while (controller.keepTraining(0.0)) {
			train(userFeatures, itemFeatures);
			calculateStatistics(userFeatures, itemFeatures);
		}

		return new PureSVDModel(ImmutableMatrix.wrap(userFeatures),
				ImmutableMatrix.wrap(itemFeatures),
				snapshot.userIndex(), snapshot.itemIndex(),
				featureInfo);
	}

	private void train(Matrix userFeatures, Matrix itemFeatures) {
		LongCollection itemIdCollection = snapshot.getItemIds();
		LongCollection userIdCollection = snapshot.getUserIds();
		for (long userId : userIdCollection) {
			Collection<IndexedPreference> preferences = snapshot.getUserRatings(userId);
			Map<Long, Double> userRatings = new HashMap<Long, Double>();
			for (IndexedPreference preference : preferences) {
				userRatings.put(preference.getItemId(), preference.getValue());
			}
			for (long itemId : itemIdCollection) {
				double rating = 0.0;
				if (userRatings.containsKey(itemId)) {
					rating = userRatings.get(itemId);
				}
				trainFeatures(userFeatures, itemFeatures, itemId, userId, rating);
			}
		}
	}

	private void trainFeatures(Matrix userFeatures, Matrix itemFeatures, long itemId, long userId, double rating) {
		int itemIndex = snapshot.itemIndex().getIndex(itemId);
		int userIndex = snapshot.userIndex().getIndex(userId);
		AVector item = itemFeatures.getRow(itemIndex);
		AVector user = userFeatures.getRow(userIndex);
		double prediction = item.dotProduct(user);
		double error = rating - prediction;
		if (Double.isNaN(error) || Double.isInfinite(error)) {
			System.out.printf("Error is " + error);
		}
		for (int i = 0; i < featureCount; i++) {
			double val = item.get(i) + learningRate * (2 * error * user.get(i) - regularization * item.get(i));
			if (Double.isNaN(val) || Double.isInfinite(val)) {
				System.out.println("Val is " + val);
			} else {
				item.set(i, val);
			}

			val = user.get(i) + learningRate * (2 * error * item.get(i) - regularization * user.get(i));
			if (Double.isNaN(val) || Double.isInfinite(val)) {
				System.out.println("Val is " + val);
			} else {
				user.set(i, val);
			}
		}
	}

	private void calculateStatistics(Matrix userFeatures, Matrix itemFeatures) {
		double sum = 0;
		LongCollection itemIdCollection = snapshot.getItemIds();
		LongCollection userIdCollection = snapshot.getUserIds();
		for (long userId : userIdCollection) {
			Collection<IndexedPreference> preferences = snapshot.getUserRatings(userId);
			Map<Long, Double> userRatings = new HashMap<Long, Double>();
			for (IndexedPreference preference : preferences) {
				userRatings.put(preference.getItemId(), preference.getValue());
			}
			for (long itemId : itemIdCollection) {
				double rating = 0.0;
				if (userRatings.containsKey(itemId)) {
					rating = userRatings.get(itemId);
				}
				sum += calculateStatisticsForItems(userFeatures, itemFeatures, itemId, userId, rating);
			}
		}
		System.out.println("MAE " + sum / userIdCollection.size() / itemIdCollection.size());
	}

	private double calculateStatisticsForItems(Matrix userFeatures, Matrix itemFeatures, long itemId, long userId, double rating) {
		int itemIndex = snapshot.itemIndex().getIndex(itemId);
		int userIndex = snapshot.userIndex().getIndex(userId);
		AVector item = itemFeatures.getRow(itemIndex);
		AVector user = userFeatures.getRow(userIndex);
		double prediction = item.dotProduct(user);
		double error = rating - prediction;

		return Math.abs(error);
	}
}
