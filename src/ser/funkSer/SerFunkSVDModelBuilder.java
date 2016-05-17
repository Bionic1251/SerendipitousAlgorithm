package ser.funkSer;

import annotation.RatingPredictor2;
import evaluationMetric.Container;
import mikera.matrixx.Matrix;
import mikera.matrixx.impl.ImmutableMatrix;
import mikera.vectorz.AVector;
import mikera.vectorz.Vector;
import org.apache.commons.lang3.time.StopWatch;
import org.grouplens.lenskit.ItemScorer;
import org.grouplens.lenskit.core.Transient;
import org.grouplens.lenskit.data.pref.IndexedPreference;
import org.grouplens.lenskit.data.snapshot.PreferenceSnapshot;
import org.grouplens.lenskit.iterative.TrainingLoopController;
import org.grouplens.lenskit.mf.funksvd.*;
import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import pop.PopModel;
import util.Settings;

import javax.annotation.Nonnull;
import javax.inject.Inject;
import javax.inject.Provider;
import java.util.*;

public class SerFunkSVDModelBuilder implements Provider<SerFunkSVDModel> {
	private static Logger logger = LoggerFactory.getLogger(FunkSVDModelBuilder.class);

	protected final int featureCount;
	protected final PreferenceSnapshot snapshot;
	protected final double initialValue;
	private final Map<Long, Double> obviousMap = new HashMap<Long, Double>();

	protected final SerFunkSVDUpdateRule rule;

	@Inject
	public SerFunkSVDModelBuilder(@Transient @Nonnull PreferenceSnapshot snapshot,
							   @Transient @Nonnull SerFunkSVDUpdateRule rule,
							   @FeatureCount int featureCount,
							   @InitialFeatureValue double initVal, @RatingPredictor2 ItemScorer obviousItemScorer, PopModel popModel) {
		this.featureCount = featureCount;
		this.initialValue = initVal;
		this.snapshot = snapshot;
		this.rule = rule;
		fillObviousMap(obviousItemScorer);
		List<Long> list = new ArrayList<Long>(popModel.getItemList());
		Collections.reverse(list);
		Set<Long> itemSet = new HashSet<Long>(list.subList(0, Settings.POPULAR_ITEMS_SERENDIPITY_NUMBER));
		for (Long itemId : itemSet) {
			obviousMap.put(itemId, (double) snapshot.getUserIds().size());
		}
	}

	private void fillObviousMap(ItemScorer obviousItemScorer) {
		Collection<Long> userIds = snapshot.getUserIds();
		for (Long userId : userIds) {
			MutableSparseVector vector = MutableSparseVector.create(snapshot.getItemIds());
			obviousItemScorer.score(userId, vector);
			Set<Long> expectedSet = getExpectedSet(userId, vector);
			for (Long itemId : expectedSet) {
				addToMap(itemId);
			}
		}
	}

	private double getWeight(Long itemId) {
		if (!obviousMap.containsKey(itemId)) {
			return 1.0;
		}
		double val = obviousMap.get(itemId);
		double size = snapshot.getUserIds().size();
		double w = 1 - val / size / 4.0;
		return w;
	}

	private void addToMap(Long itemId) {
		double score = 0;
		if (obviousMap.containsKey(itemId)) {
			score = obviousMap.get(itemId);
		}
		score += 1;
		obviousMap.put(itemId, score);
	}

	private Set<Long> getExpectedSet(long user, MutableSparseVector scores) {
		List<Container<Double>> list = new ArrayList<Container<Double>>();
		Set<Long> expectedSet = new HashSet<Long>();
		for (Long key : scores.keySet()) {
			list.add(new Container<Double>(key, scores.get(key)));
		}
		Collections.sort(list);
		Collections.reverse(list);
		Set<Long> trainingSet = new HashSet<Long>();
		Collection<IndexedPreference> prefs = snapshot.getUserRatings(user);
		for (IndexedPreference pref : prefs) {
			trainingSet.add(pref.getItemId());
		}
		int i = 0;
		while (expectedSet.size() < Settings.ADDITIONAL_OBVIOUS) {
			Long id = list.get(i).getId();
			if (!trainingSet.contains(id)) {
				expectedSet.add(id);
			}
			i++;
		}
		return expectedSet;
	}


	@Override
	public SerFunkSVDModel get() {
		int userCount = snapshot.getUserIds().size();
		Matrix userFeatures = Matrix.create(userCount, featureCount);

		int itemCount = snapshot.getItemIds().size();
		Matrix itemFeatures = Matrix.create(itemCount, featureCount);

		logger.debug("Learning rate is {}", rule.getLearningRate());
		logger.debug("Regularization term is {}", rule.getTrainingRegularization());

		logger.info("Building SVD with {} features for {} ratings",
				featureCount, snapshot.getRatings().size());

		SerTrainingEstimator estimates = rule.makeEstimator(snapshot);

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
		return new SerFunkSVDModel(ImmutableMatrix.wrap(userFeatures),
				ImmutableMatrix.wrap(itemFeatures),
				snapshot.userIndex(), snapshot.itemIndex(),
				featureInfo);
	}

	/**
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
	 * @see #summarizeFeature(mikera.vectorz.AVector, mikera.vectorz.AVector, FeatureInfo.Builder)
	 */
	protected void trainFeature(int feature, SerTrainingEstimator estimates,
								Vector userFeatureVector, Vector itemFeatureVector,
								FeatureInfo.Builder fib) {
		double rmse = Double.MAX_VALUE;
		double trail = initialValue * initialValue * (featureCount - feature - 1);
		TrainingLoopController controller = rule.getTrainingLoopController();
		Collection<IndexedPreference> ratings = snapshot.getRatings();
		while (controller.keepTraining(rmse)) {
			rmse = doFeatureIteration(estimates, ratings, userFeatureVector, itemFeatureVector, trail);
			fib.addTrainingRound(rmse);
			logger.trace("iteration {} finished with RMSE {}", controller.getIterationCount(), rmse);
		}
	}

	/**
	 * Do a single feature iteration.
	 *
	 * @param estimates         The estimates.
	 * @param ratings           The ratings to train on.
	 * @param userFeatureVector The user column vector for the current feature.
	 * @param itemFeatureVector The item column vector for the current feature.
	 * @param trail             The sum of the remaining user-item-feature values.
	 * @return The RMSE of the feature iteration.
	 */
	protected double doFeatureIteration(SerTrainingEstimator estimates,
										Collection<IndexedPreference> ratings,
										Vector userFeatureVector, Vector itemFeatureVector,
										double trail) {
		// We'll create a fresh updater for each feature iteration
		// Not much overhead, and prevents needing another parameter
		SerFunkSVDUpdater updater = rule.createUpdater();

		for (IndexedPreference r : ratings) {
			final int uidx = r.getUserIndex();
			final int iidx = r.getItemIndex();

			updater.prepare(0, r.getValue(), estimates.get(r),
					userFeatureVector.get(uidx), itemFeatureVector.get(iidx), trail, getWeight(r.getItemId()));

			// Step 3: Update feature values
			userFeatureVector.addAt(uidx, updater.getUserFeatureUpdate());
			itemFeatureVector.addAt(iidx, updater.getItemFeatureUpdate());
		}

		return updater.getRMSE();
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
