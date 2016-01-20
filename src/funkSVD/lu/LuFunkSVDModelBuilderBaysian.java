package funkSVD.lu;

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
import org.grouplens.lenskit.mf.funksvd.FeatureCount;
import org.grouplens.lenskit.mf.funksvd.FeatureInfo;
import org.grouplens.lenskit.mf.funksvd.InitialFeatureValue;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import javax.inject.Inject;
import javax.inject.Provider;
import java.util.*;

/**
 * Baseline recommender builder using gradient descent (Funk Baseline).
 * <p/>
 * <p>
 * This recommender builder constructs an Baseline-based recommender using gradient
 * descent, as pioneered by Simon Funk.  It also incorporates the regularizations
 * Funk did. These are documented in
 * <a href="http://sifter.org/~simon/journal/20061211.html">Netflix Update: Try
 * This at Home</a>. This implementation is based in part on
 * <a href="http://www.timelydevelopment.com/demos/NetflixPrize.aspx">Timely
 * Development's sample code</a>.</p>
 *
 * @author <a href="http://www.grouplens.org">GroupLens Research</a>
 */
public class LuFunkSVDModelBuilderBaysian implements Provider<LuFunkSVDModelBaysian> {
	private static Logger logger = LoggerFactory.getLogger(LuFunkSVDModelBuilderBaysian.class);
	private int count;
	private double func;
	private final int RANGE = 10000;

	protected final int featureCount;
	protected final PreferenceSnapshot snapshot;
	protected final double initialValue;
	protected final Map<Integer, Double> popMap = new HashMap<Integer, Double>();
	private final double threshold;
	private final PreferenceDomain domain;
	private final double alpha;

	protected final LuFunkSVDUpdateRule rule;

	@Inject
	public LuFunkSVDModelBuilderBaysian(@Transient @Nonnull PreferenceSnapshot snapshot,
										@Transient @Nonnull LuFunkSVDUpdateRule rule,
										@FeatureCount int featureCount,
										@InitialFeatureValue double initVal, @Threshold double threshold, @Nullable PreferenceDomain dom,
										@Alpha double alpha) {
		this.featureCount = featureCount;
		this.initialValue = initVal;
		this.snapshot = snapshot;
		this.rule = rule;
		this.threshold = threshold;
		domain = dom;
		this.alpha = alpha;
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
	public LuFunkSVDModelBaysian get() {
		populatePopMap();

		int userCount = snapshot.getUserIds().size();
		Matrix userFeatures = Matrix.create(userCount, featureCount);

		int itemCount = snapshot.getItemIds().size();
		Matrix itemFeatures = Matrix.create(itemCount, featureCount);

		logger.debug("Learning rate is {}", rule.getLearningRate());
		logger.debug("Regularization term is {}", rule.getTrainingRegularization());

		logger.info("Building Baseline with {} features for {} ratings",
				featureCount, snapshot.getRatings().size());

		LuTrainingEstimator estimates = rule.makeEstimator(snapshot);

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
		return new LuFunkSVDModelBaysian(ImmutableMatrix.wrap(userFeatures),
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
	protected void trainFeature(int feature, LuTrainingEstimator estimates,
								Vector userFeatureVector, Vector itemFeatureVector,
								FeatureInfo.Builder fib) {
		System.out.println("Feature " + feature);
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
	protected double doFeatureIteration(LuTrainingEstimator estimates,
										Collection<IndexedPreference> ratings,
										Vector userFeatureVector, Vector itemFeatureVector,
										double trail) {
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
					trainPair(userFeatureVector, itemFeatureVector, liked, disliked, estimates, trail);
				}
			}
		}


		System.out.println("Pairs count: " + count + "; Function value: " + func / count);
		return 0.0;
	}

	private void trainPair(Vector userFeatureVector, Vector itemFeatureVector, IndexedPreference liked, IndexedPreference disliked, LuTrainingEstimator estimates, double trail) {
		double uv = userFeatureVector.get(liked.getUserIndex());
		double likedIV = itemFeatureVector.get(liked.getItemIndex());
		double likedPred = estimates.get(liked) + uv * likedIV + trail;
		double dislikedIV = itemFeatureVector.get(disliked.getItemIndex());
		double dislikedPred = estimates.get(disliked) + uv * dislikedIV + trail;
		//double rawDiff = likedPred - dislikedPred;

		dislikedPred = domain.clampValue(dislikedPred);
		likedPred = domain.clampValue(likedPred);

		double pop = Math.pow(popMap.get(disliked.getItemIndex()) + 1, alpha);
		double diff = likedPred - dislikedPred;
		func += function(diff, pop);
		if (diff > 0) {
			count++;
			return;
		}
		/*if (rawDiff > domain.getMaximum() - domain.getMinimum()) {
			return;
		}*/

		double itemDerivativeVal = getDerivative(uv, pop, diff);
		double userDerivativeVal = getDerivative(likedIV, pop, diff);

		double updateItemVal = (itemDerivativeVal - rule.getTrainingRegularization() * likedIV) * rule.getLearningRate();
		double updateDisItemVal = (-itemDerivativeVal - rule.getTrainingRegularization() * dislikedIV) * rule.getLearningRate();
		double updateUserVal = (userDerivativeVal - rule.getTrainingRegularization() * uv) * rule.getLearningRate();

		double updatedLikeItemValue = updateItemVal + likedIV;
		double updatedDisItemValue = updateItemVal + dislikedIV;
		double updatedUserValue = updateUserVal + uv;

		if (!Double.isNaN(updateItemVal) && !Double.isInfinite(updateItemVal) && isInRange(updatedLikeItemValue)) {
			itemFeatureVector.addAt(liked.getItemIndex(), updateItemVal);
		}

		if (!Double.isNaN(updateUserVal) && !Double.isInfinite(updateUserVal) && isInRange(updatedUserValue)) {
			userFeatureVector.addAt(liked.getUserIndex(), updateUserVal);
		}

		if (!Double.isNaN(updateDisItemVal) && !Double.isInfinite(updateDisItemVal) && isInRange(updatedDisItemValue)) {
			itemFeatureVector.addAt(disliked.getItemIndex(), updateDisItemVal);
		}
	}

	protected double getDerivative(double a, double pop, double diff) {
		return a * Math.exp(diff) / (1 + Math.exp(diff)) * pop;
	}

	protected double function(double diff, double pop) {
		return Math.log(1 + Math.exp(diff)) * pop;
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
