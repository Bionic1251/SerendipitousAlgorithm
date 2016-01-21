package funkSVD.zheng;

import annotation.Alpha;
import annotation.Threshold;
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
import org.grouplens.lenskit.iterative.TrainingLoopController;
import org.grouplens.lenskit.knn.item.model.ItemItemModel;
import org.grouplens.lenskit.mf.funksvd.FeatureCount;
import org.grouplens.lenskit.mf.funksvd.FeatureInfo;
import org.grouplens.lenskit.mf.funksvd.InitialFeatureValue;
import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.grouplens.lenskit.vectors.SparseVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import javax.inject.Inject;
import javax.inject.Provider;
import java.util.*;

/**
 * baseline recommender builder using gradient descent (Funk baseline).
 * <p/>
 * <p>
 * This recommender builder constructs an baseline-based recommender using gradient
 * descent, as pioneered by Simon Funk.  It also incorporates the regularizations
 * Funk did. These are documented in
 * <a href="http://sifter.org/~simon/journal/20061211.html">Netflix Update: Try
 * This at Home</a>. This implementation is based in part on
 * <a href="http://www.timelydevelopment.com/demos/NetflixPrize.aspx">Timely
 * Development's sample code</a>.</p>
 *
 * @author <a href="http://www.grouplens.org">GroupLens Research</a>
 */
public class ZhengFunkSVDModelBuilder implements Provider<ZhengFunkSVDModel> {
	private static Logger logger = LoggerFactory.getLogger(ZhengFunkSVDModelBuilder.class);
	private int count;
	private double func;

	protected final int featureCount;
	protected final PreferenceSnapshot snapshot;
	protected final double initialValue;
	private final ItemItemModel itemItemModel;
	private final Map<Integer, Double> popMap = new HashMap<Integer, Double>();
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
		populateUserItemMap();
		populatePopMap();

		int userCount = snapshot.getUserIds().size();
		Matrix userFeatures = Matrix.create(userCount, featureCount);

		int itemCount = snapshot.getItemIds().size();
		Matrix itemFeatures = Matrix.create(itemCount, featureCount);

		logger.debug("Learning rate is {}", rule.getLearningRate());
		logger.debug("Regularization term is {}", rule.getTrainingRegularization());

		logger.info("Building baseline with {} features for {} ratings",
				featureCount, snapshot.getRatings().size());

		ZhengTrainingEstimator estimates = rule.makeEstimator(snapshot);

		List<FeatureInfo> featureInfo = new ArrayList<FeatureInfo>(featureCount);

		// Use scratch vectors for each feature for better cache locality
		// Per-feature vectors are strided in the output matrices
		Vector uvec = Vector.createLength(userCount);
		Vector ivec = Vector.createLength(itemCount);

		for (int f = 0; f < featureCount; f++) {
			logger.debug("Training feature {}", f);
			StopWatch timer = new StopWatch();
			timer.start();

			uvec.fill(initialValue);
			ivec.fill(initialValue);

			FeatureInfo.Builder fib = new FeatureInfo.Builder(f);
			trainFeature(f, estimates, uvec, ivec, fib);
			summarizeFeature(uvec, ivec, fib);
			featureInfo.add(fib.build());

			// Update each rating's cached value to accommodate the feature values.
			estimates.update(uvec, ivec);

			// And store the data into the matrix
			userFeatures.setColumn(f, uvec);
			assert Math.abs(userFeatures.getColumnView(f).elementSum() - uvec.elementSum()) < 1.0e-4 : "user column sum matches";
			itemFeatures.setColumn(f, ivec);
			assert Math.abs(itemFeatures.getColumnView(f).elementSum() - ivec.elementSum()) < 1.0e-4 : "item column sum matches";

			timer.stop();
			logger.info("Finished feature {} in {}", f, timer);
		}

		// Wrap the user/item matrices because we won't use or modify them again
		return new ZhengFunkSVDModel(ImmutableMatrix.wrap(userFeatures),
				ImmutableMatrix.wrap(itemFeatures),
				snapshot.userIndex(), snapshot.itemIndex(),
				featureInfo);
	}

	/**
	 * Train a feature using a collection of ratings.  This method iteratively calls {@link
	 * #doFeatureIteration  to train
	 * the feature.  It can be overridden to customize the feature training strategy.
	 * <p/>
	 * <p>We use the estimator to maintain the estimate up through a particular feature value,
	 * rather than recomputing the entire kernel value every time.  This hopefully speeds up training.
	 * It means that we always tell the updater we are training feature 0, but use a subvector that
	 * starts with the current feature.</p>
	 *
	 * @param feature           The number of the current feature.
	 * @param estimates         The current estimator.  This method is <b>not</b> expected to update the
	 *                          estimator.
	 * @param userFeatureVector The user feature values.  This has been initialized to the initial value,
	 *                          and may be reused between features.
	 * @param itemFeatureVector The item feature values.  This has been initialized to the initial value,
	 *                          and may be reused between features.
	 * @param fib               The feature info builder. This method is only expected to add information
	 *                          about its training rounds to the builder; the caller takes care of feature
	 *                          number and summary data.
	 * @see #doFeatureIteration(TrainingEstimator, Collection, Vector, Vector, double)
	 * @see #summarizeFeature(AVector, AVector, FeatureInfo.Builder)
	 */
	protected void trainFeature(int feature, ZhengTrainingEstimator estimates,
								Vector userFeatureVector, Vector itemFeatureVector,
								FeatureInfo.Builder fib) {
		double trail = initialValue * initialValue * (featureCount - feature - 1);
		TrainingLoopController controller = rule.getTrainingLoopController();
		while (controller.keepTraining(0.0)) {
			doFeatureIteration(estimates, userFeatureVector, itemFeatureVector, trail);
			fib.addTrainingRound(0.0);
		}
	}

	/**
	 * Do a single feature iteration.
	 *
	 * @param estimates         The estimates.
	 * @param userFeatureVector The user column vector for the current feature.
	 * @param itemFeatureVector The item column vector for the current feature.
	 * @param trail             The sum of the remaining user-item-feature values.
	 * @return The RMSE of the feature iteration.
	 */
	protected void doFeatureIteration(ZhengTrainingEstimator estimates,
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
				sum += trainFeatures(itemFeatureVector, userFeatureVector, itemId, userId, rating, estimates, trail);
			}
		}
		System.out.println("MAE " + sum / userIdCollection.size() / itemIdCollection.size());
	}

	private double trainFeatures(Vector item, Vector user, long itemId, long userId, double rating, ZhengTrainingEstimator estimates, double trail) {
		int itemIndex = snapshot.itemIndex().getIndex(itemId);
		int userIndex = snapshot.userIndex().getIndex(userId);
		double prediction = estimates.get(userId, itemId) + item.get(itemIndex) * user.get(userIndex) + trail;
		prediction = domain.clampValue(prediction);
		double dissimilarity = 0.0;
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

		if (!Double.isNaN(updateItemVal) && !Double.isInfinite(updateItemVal) && isInRange(updatedItemValue)) {
			item.addAt(itemIndex, updateItemVal);
		}

		if (!Double.isNaN(updateUserVal) && !Double.isInfinite(updateUserVal) && isInRange(updatedUserValue)) {
			user.addAt(userIndex, updateUserVal);
		}

		return Math.abs(error);
	}

	private boolean isInRange(double value){
		return value < RANGE && value > -RANGE;
	}

	/**
	 * Add a feature's summary to the feature info builder.
	 *
	 * @param ufv The user values.
	 * @param ifv The item values.
	 * @param fib The feature info builder.
	 */
	protected void summarizeFeature(AVector ufv, AVector ifv, FeatureInfo.Builder fib) {
		fib.setUserAverage(ufv.elementSum() / ufv.length())
				.setItemAverage(ifv.elementSum() / ifv.length())
				.setSingularValue(ufv.magnitude() * ifv.magnitude());
	}
}
