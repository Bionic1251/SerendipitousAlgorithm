package lu;

import it.unimi.dsi.fastutil.longs.LongCollection;
import mikera.matrixx.Matrix;
import mikera.matrixx.impl.ImmutableMatrix;
import mikera.vectorz.AVector;
import mikera.vectorz.Vector;
import org.grouplens.lenskit.core.Transient;
import org.grouplens.lenskit.data.pref.IndexedPreference;
import org.grouplens.lenskit.data.snapshot.PreferenceSnapshot;
import org.grouplens.lenskit.iterative.TrainingLoopController;
import org.grouplens.lenskit.mf.funksvd.FeatureCount;
import org.grouplens.lenskit.mf.funksvd.FeatureInfo;
import org.grouplens.lenskit.mf.funksvd.InitialFeatureValue;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import spr2.TestTrainingEstimator;

import javax.annotation.Nonnull;
import javax.inject.Inject;
import javax.inject.Provider;
import java.util.*;

/**
 * SVD recommender builder using gradient descent (Funk SVD).
 * <p/>
 * <p>
 * This recommender builder constructs an SVD-based recommender using gradient
 * descent, as pioneered by Simon Funk.  It also incorporates the regularizations
 * Funk did. These are documented in
 * <a href="http://sifter.org/~simon/journal/20061211.html">Netflix Update: Try
 * This at Home</a>. This implementation is based in part on
 * <a href="http://www.timelydevelopment.com/demos/NetflixPrize.aspx">Timely
 * Development's sample code</a>.</p>
 *
 * @author <a href="http://www.grouplens.org">GroupLens Research</a>
 */
public class LuFunkSVDModelBuilder implements Provider<LuFunkSVDModel> {
	private static Logger logger = LoggerFactory.getLogger(LuFunkSVDModelBuilder.class);

	private final double ALPHA = 0.5;
	private final double MAX_VALUE = 5.0;
	private final double MIN_VALUE = 0.0;

	protected final int featureCount;
	protected final PreferenceSnapshot snapshot;
	protected final double initialValue;
	protected final Map<Integer, Double> popMap = new HashMap<Integer, Double>();

	protected final LuFunkSVDUpdateRule rule;

	@Inject
	public LuFunkSVDModelBuilder(@Transient @Nonnull PreferenceSnapshot snapshot,
								 @Transient @Nonnull LuFunkSVDUpdateRule rule,
								 @FeatureCount int featureCount,
								 @InitialFeatureValue double initVal) {
		this.featureCount = featureCount;
		this.initialValue = initVal;
		this.snapshot = snapshot;
		this.rule = rule;
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
	public LuFunkSVDModel get() {
		populatePopMap();

		int userCount = snapshot.getUserIds().size();
		Matrix userFeatures = Matrix.create(userCount, featureCount);

		int itemCount = snapshot.getItemIds().size();
		Matrix itemFeatures = Matrix.create(itemCount, featureCount);

		logger.debug("Learning rate is {}", rule.getLearningRate());
		logger.debug("Regularization term is {}", rule.getTrainingRegularization());

		logger.info("Building SVD with {} features for {} ratings",
				featureCount, snapshot.getRatings().size());

		LuTrainingEstimator estimates = rule.makeEstimator(snapshot);

		List<FeatureInfo> featureInfo = new ArrayList<FeatureInfo>(featureCount);

		// Use scratch vectors for each feature for better cache locality
		// Per-feature vectors are strided in the output matrices
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

		for (int k = 0; k < 5; k++) {
			int count = 0;
			double func = 0;
			double alpha = rule.getLearningRate();
			double betta = 0;// rule.getTrainingRegularization();
			LongCollection userIds = snapshot.getUserIds();
			for (long usedId : userIds) {
				Collection<IndexedPreference> userRatings = snapshot.getUserRatings(usedId);
				for (IndexedPreference liked : userRatings) {
					if (liked.getValue() <= 3.0) {
						continue;
					}
					for (IndexedPreference disliked : userRatings) {
						if (disliked.getValue() > 3.0) {
							continue;
						}


						AVector user = userFeatures.getRow(liked.getUserIndex());
						AVector likedVec = itemFeatures.getRow(liked.getItemIndex());
						AVector dislikedVec = itemFeatures.getRow(disliked.getItemIndex());
						double likedPred = likedVec.dotProduct(user);
						likedPred = getClappedValue(likedPred);
						double dislikedPred = dislikedVec.dotProduct(user);
						dislikedPred = getClappedValue(dislikedPred);

						double diff = likedPred - dislikedPred;
						func += Math.log(1 + Math.exp(diff));
						if (diff <= 0) {
							count++;
						}

						if(diff >= MAX_VALUE - MIN_VALUE){
							continue;
						}

						if (Double.isNaN(diff) || Double.isInfinite(diff)) {
							System.out.println("Yo");
						}

						double pop = Math.pow(popMap.get(disliked.getItemIndex()) + 1, ALPHA);
						//pop = 1;
						for (int i = 0; i < featureCount; i++) {
							double der = user.get(i) * Math.exp(diff) / (1 + Math.exp(diff)) * pop;
							double val = likedVec.get(i) + alpha * (der - betta * likedVec.get(i));
							if (Double.isNaN(val) || Double.isInfinite(val)) {
								System.out.println("Yo");
							}
							likedVec.set(i, val);

							der = -user.get(i) * Math.exp(diff) / (1 + Math.exp(diff)) * pop;
							val = dislikedVec.get(i) + alpha * (der - betta * dislikedVec.get(i));
							if (Double.isNaN(val) || Double.isInfinite(val)) {
								System.out.println("Yo");
							}
							dislikedVec.set(i, val);

							der = (likedVec.get(i) - dislikedVec.get(i)) * Math.exp(diff) / (1 + Math.exp(diff)) * pop;
							val = user.get(i) + alpha * (der - betta * user.get(i));
							if (Double.isNaN(val) || Double.isInfinite(val)) {
								System.out.println("Yo");
							}
							user.set(i, val);
						}
					}
				}
			}
			System.out.println("Inc " + count + " Error " + func);
		}


		// Wrap the user/item matrices because we won't use or modify them again
		return new LuFunkSVDModel(ImmutableMatrix.wrap(userFeatures),
				ImmutableMatrix.wrap(itemFeatures),
				snapshot.userIndex(), snapshot.itemIndex(),
				featureInfo);
	}

	private double getClappedValue(double value) {
		if (value > MAX_VALUE) {
			return MAX_VALUE;
		} else if (value < MIN_VALUE) {
			return MIN_VALUE;
		}
		return value;
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
	protected void trainFeature(int feature, TestTrainingEstimator estimates,
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
	protected double doFeatureIteration(TestTrainingEstimator estimates,
										Collection<IndexedPreference> ratings,
										Vector userFeatureVector, Vector itemFeatureVector,
										double trail) {
		int count = 0;
		LongCollection userIds = snapshot.getUserIds();
		for (long usedId : userIds) {
			Collection<IndexedPreference> userRatings = snapshot.getUserRatings(usedId);
			for (IndexedPreference liked : userRatings) {
				if (liked.getValue() <= 3.0) {
					continue;
				}
				for (IndexedPreference disliked : userRatings) {
					if (disliked.getValue() > 3.0) {
						continue;
					}

					double uv = userFeatureVector.get(liked.getUserIndex());
					double likedIV = itemFeatureVector.get(liked.getItemIndex());
					double likedPred = estimates.get(liked) + uv * likedIV + trail;

					double dislikedIV = itemFeatureVector.get(disliked.getItemIndex());
					double dislikedPred = estimates.get(disliked) + uv * dislikedIV + trail;

					double diff = likedPred - dislikedPred;
					if (diff < 0) {
						count++;
					}
					double pop = Math.pow(popMap.get(disliked.getItemIndex()) + 1, ALPHA) * 10;
					double itemDerivativeVal = -uv * Math.exp(diff) / (Math.exp(diff) + 1) * pop;
					double userDerivativeVal = -likedIV * Math.exp(diff) / (Math.exp(diff) + 1) * pop;

					double updateItemVal = (-itemDerivativeVal - rule.getTrainingRegularization() * likedIV) * rule.getLearningRate();
					double updateUserVal = (-userDerivativeVal - rule.getTrainingRegularization() * uv) * rule.getLearningRate();

					userFeatureVector.addAt(liked.getUserIndex(), updateUserVal);
					itemFeatureVector.addAt(liked.getItemIndex(), updateItemVal);
				}
			}
		}

		// We'll create a fresh updater for each feature iteration
		// Not much overhead, and prevents needing another parameter
		/*SPRFunkSVDUpdater updater = rule.createUpdater();

		for (IndexedPreference r : ratings) {
			final int uidx = r.getUserIndex();
			final int iidx = r.getItemIndex();

			updater.prepare(r.getValue(), estimates.get(r),
					userFeatureVector.get(uidx), itemFeatureVector.get(iidx), trail);

			// Step 3: Update feature values
			userFeatureVector.addAt(uidx, updater.getUserFeatureUpdate());
			itemFeatureVector.addAt(iidx, updater.getItemFeatureUpdate());
		}

		return updater.getRMSE();*/
		System.out.println("inc " + count);
		return 0.0;
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
