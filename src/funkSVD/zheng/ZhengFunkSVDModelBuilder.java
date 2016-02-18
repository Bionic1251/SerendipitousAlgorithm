package funkSVD.zheng;

import it.unimi.dsi.fastutil.longs.LongCollection;
import mf.zheng.ZhengSVDModelBuilder;
import mikera.matrixx.Matrix;
import mikera.matrixx.impl.ImmutableMatrix;
import mikera.vectorz.AVector;
import mikera.vectorz.Vector;
import org.apache.commons.lang3.time.StopWatch;
import org.grouplens.lenskit.core.Transient;
import org.grouplens.lenskit.data.pref.IndexedPreference;
import org.grouplens.lenskit.data.pref.PreferenceDomain;
import org.grouplens.lenskit.data.snapshot.PreferenceSnapshot;
import org.grouplens.lenskit.iterative.TrainingLoopController;
import org.grouplens.lenskit.knn.item.model.ItemItemModel;
import org.grouplens.lenskit.mf.funksvd.FeatureCount;
import org.grouplens.lenskit.mf.funksvd.FeatureInfo;
import org.grouplens.lenskit.mf.funksvd.InitialFeatureValue;
import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.grouplens.lenskit.vectors.SparseVector;
import util.Util;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import javax.inject.Inject;
import javax.inject.Provider;
import java.util.*;

public class ZhengFunkSVDModelBuilder implements Provider<ZhengFunkSVDModel> {

	protected final int featureCount;
	protected final PreferenceSnapshot snapshot;
	protected final double initialValue;
	private final ItemItemModel itemItemModel;
	private Map<Integer, Double> popMap = new HashMap<Integer, Double>();
	private final Map<Long, SparseVector> userItemDissimilarityMap = new HashMap<Long, SparseVector>();
	private final PreferenceDomain domain;
	private final int RANGE = 10000;

	protected final ZhengFunkSVDUpdateRule rule;

	@Inject
	public ZhengFunkSVDModelBuilder(@Transient @Nonnull PreferenceSnapshot snapshot,
									@Transient @Nonnull ZhengFunkSVDUpdateRule rule,
									@FeatureCount int featureCount,
									@InitialFeatureValue double initVal, @Nullable PreferenceDomain dom,
									ItemItemModel itemItemModel) {
		this.featureCount = featureCount;
		this.initialValue = initVal;
		this.snapshot = snapshot;
		this.rule = rule;
		domain = dom;
		this.itemItemModel = itemItemModel;
	}


	private void populatePopMap() {
		Collection<IndexedPreference> ratings = snapshot.getRatings();
		for (IndexedPreference pref : ratings) {
			double val = 0.0;
			if (popMap.containsKey(pref.getItemIndex())) {
				val = popMap.get(pref.getItemIndex());
			}
			val++;
			popMap.put(pref.getItemIndex(), val);
		}
		for (Integer key : popMap.keySet()) {
			double val = popMap.get(key);
			val /= snapshot.getUserIds().size();
			popMap.put(key, val);
		}
	}

	private void populateUserItemMap() {
		LongCollection userIds = snapshot.getUserIds();
		LongCollection itemIds = snapshot.getItemIds();
		int size = userIds.size();
		int maxCount = 0;
		for (long userId : userIds) {
			MutableSparseVector itemDissimilarityVector = MutableSparseVector.create(itemIds, 0.0);
			Collection<IndexedPreference> ratings = snapshot.getUserRatings(userId);
			for (long itemId : itemIds) {
				double dissimilaritySum = 0;
				int count = 0;
				for (IndexedPreference rating : ratings) {
					SparseVector vector = itemItemModel.getNeighbors(itemId);
					if (rating.getItemId() == itemId || !vector.containsKey(rating.getItemId())) {
						continue;
					}
					dissimilaritySum += 1 - vector.get(rating.getItemId());
					count++;
				}
				maxCount = Math.max(maxCount, count);
				if (count == 0) {
					continue;
				}
				dissimilaritySum = dissimilaritySum / count;
				if (Double.isNaN(dissimilaritySum)) {
					System.out.println(count + " !!!");
				}
				itemDissimilarityVector.set(itemId, dissimilaritySum);
			}
			userItemDissimilarityMap.put(userId, itemDissimilarityVector);
			size--;
			if (size % 100 == 0) {
				System.out.println(size + " left");
			}
		}
		System.out.println("Count " + maxCount);
	}

	@Override
	public ZhengFunkSVDModel get() {
		System.out.println(ZhengSVDModelBuilder.class);
		populateUserItemMap();
		popMap = Util.getPopMap(snapshot);

		int userCount = snapshot.getUserIds().size();
		Matrix userFeatures = Matrix.create(userCount, featureCount);

		int itemCount = snapshot.getItemIds().size();
		Matrix itemFeatures = Matrix.create(itemCount, featureCount);

		ZhengTrainingEstimator estimates = rule.makeEstimator(snapshot);

		List<FeatureInfo> featureInfo = new ArrayList<FeatureInfo>(featureCount);

		Vector uvec = Vector.createLength(userCount);
		Vector ivec = Vector.createLength(itemCount);

		for (int f = 0; f < featureCount; f++) {
			System.out.println("Feature " + f);
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

		return new ZhengFunkSVDModel(ImmutableMatrix.wrap(userFeatures),
				ImmutableMatrix.wrap(itemFeatures),
				snapshot.userIndex(), snapshot.itemIndex(),
				featureInfo);
	}

	protected void trainFeature(int feature, ZhengTrainingEstimator estimates,
								Vector userFeatureVector, Vector itemFeatureVector,
								FeatureInfo.Builder fib) {
		double trail = initialValue * initialValue * (featureCount - feature - 1);
		TrainingLoopController controller = rule.getTrainingLoopController();
		calculateStatistics(estimates, userFeatureVector, itemFeatureVector, trail);
		while (controller.keepTraining(0.0)) {
			doFeatureIteration(estimates, userFeatureVector, itemFeatureVector, trail);
			calculateStatistics(estimates, userFeatureVector, itemFeatureVector, trail);
		}
	}

	protected void doFeatureIteration(ZhengTrainingEstimator estimates,
									  Vector userFeatureVector, Vector itemFeatureVector,
									  double trail) {
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
				trainFeatures(itemFeatureVector, userFeatureVector, itemId, userId, rating, estimates, trail);
			}
		}
	}

	private void trainFeatures(Vector item, Vector user, long itemId, long userId, double rating, ZhengTrainingEstimator estimates, double trail) {
		int itemIndex = snapshot.itemIndex().getIndex(itemId);
		int userIndex = snapshot.userIndex().getIndex(userId);
		double prediction = estimates.get(userId, itemId) + item.get(itemIndex) * user.get(userIndex) + trail;
		double dissimilarity = 1.0;
		if (userItemDissimilarityMap.containsKey(userId)) {
			SparseVector itemDissimilarityVector = userItemDissimilarityMap.get(userId);
			if (itemDissimilarityVector.containsKey(itemId)) {
				dissimilarity = itemDissimilarityVector.get(itemId);
			}
		}
		double w = 1 - popMap.get(itemIndex) + dissimilarity;
		double error = (rating - prediction) * w;
		if (Double.isNaN(error) || Double.isInfinite(error)) {
			System.out.printf("Error is " + error);
		}
		double learningRate = rule.getLearningRate();
		double regularization = rule.getTrainingRegularization();

		double updateItemVal = (error * user.get(userIndex) - regularization * item.get(itemIndex)) * learningRate;
		double updateUserVal = (error * item.get(itemIndex) - regularization * user.get(userIndex)) * learningRate;

		double updatedItemValue = updateItemVal + item.get(itemIndex);
		double updatedUserValue = updateUserVal + user.get(userIndex);

		if (!Double.isNaN(updatedItemValue) && !Double.isInfinite(updatedItemValue)) {
			item.addAt(itemIndex, updateItemVal);
		}

		if (!Double.isNaN(updatedUserValue) && !Double.isInfinite(updatedUserValue)) {
			user.addAt(userIndex, updateUserVal);
		}
	}

	protected void calculateStatistics(ZhengTrainingEstimator estimates,
									   Vector userFeatureVector, Vector itemFeatureVector,
									   double trail) {
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
				sum += calculateStatisticsForItem(itemFeatureVector, userFeatureVector, itemId, userId, rating, estimates, trail);
			}
		}
		System.out.println("MAE " + sum / userIdCollection.size() / itemIdCollection.size());
	}

	private double calculateStatisticsForItem(Vector item, Vector user, long itemId, long userId, double rating, ZhengTrainingEstimator estimates, double trail) {
		int itemIndex = snapshot.itemIndex().getIndex(itemId);
		int userIndex = snapshot.userIndex().getIndex(userId);
		double prediction = estimates.get(userId, itemId) + item.get(itemIndex) * user.get(userIndex) + trail;
		double dissimilarity = 1.0;
		if (userItemDissimilarityMap.containsKey(userId)) {
			SparseVector itemDissimilarityVector = userItemDissimilarityMap.get(userId);
			if (itemDissimilarityVector.containsKey(itemId)) {
				dissimilarity = itemDissimilarityVector.get(itemId);
			}
		}
		double w = 1 - popMap.get(itemIndex) + dissimilarity;
		double error = (rating - prediction) * w;

		return Math.abs(error);
	}

	protected void summarizeFeature(AVector ufv, AVector ifv, FeatureInfo.Builder fib) {
		fib.setUserAverage(ufv.elementSum() / ufv.length())
				.setItemAverage(ifv.elementSum() / ifv.length())
				.setSingularValue(ufv.magnitude() * ifv.magnitude());
	}
}
