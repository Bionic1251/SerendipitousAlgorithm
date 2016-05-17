package ser.serSVD;

import annotation.RatingPredictor2;
import evaluationMetric.Container;
import mikera.matrixx.Matrix;
import mikera.matrixx.impl.ImmutableMatrix;
import mikera.vectorz.AVector;
import mikera.vectorz.Vector;
import org.grouplens.lenskit.ItemScorer;
import org.grouplens.lenskit.core.Transient;
import org.grouplens.lenskit.data.pref.IndexedPreference;
import org.grouplens.lenskit.data.snapshot.PreferenceSnapshot;
import org.grouplens.lenskit.eval.data.traintest.TTDataSet;
import org.grouplens.lenskit.iterative.LearningRate;
import org.grouplens.lenskit.iterative.RegularizationTerm;
import org.grouplens.lenskit.iterative.StoppingCondition;
import org.grouplens.lenskit.iterative.TrainingLoopController;
import org.grouplens.lenskit.mf.funksvd.FeatureCount;
import org.grouplens.lenskit.mf.funksvd.FeatureInfo;
import org.grouplens.lenskit.mf.funksvd.InitialFeatureValue;
import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import pop.PopModel;
import util.Settings;
import util.Util;

import javax.annotation.Nonnull;
import javax.inject.Inject;
import javax.inject.Provider;
import java.util.*;

public class SerSVDModelBuilder implements Provider<SerSVDModel> {
	private static Logger logger = LoggerFactory.getLogger(SerSVDModelBuilder.class);

	protected final int featureCount;
	protected final double learningRate;
	protected final double regularization;
	private final StoppingCondition stoppingCondition;
	protected final PreferenceSnapshot snapshot;
	protected final double initialValue;
	private final Map<Long, Double> obviousMap = new HashMap<Long, Double>();

	@Inject
	public SerSVDModelBuilder(@Transient @Nonnull PreferenceSnapshot snapshot,
							  @FeatureCount int featureCount,
							  @InitialFeatureValue double initVal, @LearningRate double lrate,
							  @RegularizationTerm double reg, StoppingCondition stop, PopModel popModel, @RatingPredictor2 ItemScorer obviousItemScorer) {
		this.featureCount = featureCount;
		this.initialValue = initVal;
		this.snapshot = snapshot;
		learningRate = lrate;
		regularization = reg;
		stoppingCondition = stop;
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
			Set<Long> expectedSet = Util.getExpectedSet(userId, vector, snapshot);
			for (Long itemId : expectedSet) {
				addToMap(itemId);
			}
		}
	}

	private void addToMap(Long itemId) {
		double score = 0;
		if (obviousMap.containsKey(itemId)) {
			score = obviousMap.get(itemId);
		}
		score += 1;
		obviousMap.put(itemId, score);
	}


	@Override
	public SerSVDModel get() {
		System.out.println(SerSVDModelBuilder.class);
		int userCount = snapshot.getUserIds().size();
		Matrix userFeatures = Matrix.create(userCount, featureCount);

		int itemCount = snapshot.getItemIds().size();
		Matrix itemFeatures = Matrix.create(itemCount, featureCount);

		logger.info("Building serSVD with {} features for {} ratings",
				featureCount, snapshot.getRatings().size());

		List<FeatureInfo> featureInfo = new ArrayList<FeatureInfo>(featureCount);

		Vector uvec = Vector.createLength(userCount);
		Vector ivec = Vector.createLength(itemCount);

		Random random = new Random();
		random.setSeed(123);

		for (int f = 0; f < featureCount; f++) {
			for (int i = 0; i < userCount; i++) {
				uvec.set(i, random.nextDouble());
			}
			for (int i = 0; i < itemCount; i++) {
				//ivec.set(i, random.nextDouble());
				long id = snapshot.itemIndex().getId(i);
				if (obviousMap.containsKey(id)) {
					ivec.set(i, getWeight(id));//obviousMap.get(id) / (double) snapshot.getUserIds().size());
				} else {
					ivec.set(i, 0.01);
				}
			}
			/*uvec.fill(initialValue);
			ivec.fill(initialValue);*/

			userFeatures.setColumn(f, uvec);
			itemFeatures.setColumn(f, ivec);

			FeatureInfo.Builder fib = new FeatureInfo.Builder(f);
			featureInfo.add(fib.build());
		}

		TrainingLoopController controller = stoppingCondition.newLoop();
		calculateStatistics(userFeatures, itemFeatures);
		while (controller.keepTraining(0.0)) {
			trainFeatures(userFeatures, itemFeatures);
			calculateStatistics(userFeatures, itemFeatures);
		}

		return new SerSVDModel(ImmutableMatrix.wrap(userFeatures),
				ImmutableMatrix.wrap(itemFeatures),
				snapshot.userIndex(), snapshot.itemIndex(),
				featureInfo);
	}

	private void trainFeatures(Matrix userFeatures, Matrix itemFeatures) {
		for (IndexedPreference rating : snapshot.getRatings()) {
			double w = getWeight(rating.getItemId());
			AVector item = itemFeatures.getRow(rating.getItemIndex());
			AVector user = userFeatures.getRow(rating.getUserIndex());
			double prediction = item.dotProduct(user);
			double error = (rating.getValue() - prediction);
			for (int i = 0; i < featureCount; i++) {
				double val = item.get(i) + learningRate * (2 * error * user.get(i) - regularization * item.get(i));
				if (Double.isNaN(val) || Double.isInfinite(val)) {
					System.out.println(val);
				}
				item.set(i, val);

				val = user.get(i) + learningRate * (2 * error * item.get(i) - regularization * user.get(i));
				if (Double.isNaN(val) || Double.isInfinite(val)) {
					System.out.println(val);
				}
				user.set(i, val);
			}
		}
	}

	private double getWeight(Long itemId) {
		if (!obviousMap.containsKey(itemId)) {
			return 1.0;
		}
		double val = obviousMap.get(itemId);
		double size = snapshot.getUserIds().size();
		double w = 1.0 - val / size / 2.0;
		return w;
	}

	private void calculateStatistics(Matrix userFeatures, Matrix itemFeatures) {
		double sum = 0;
		for (IndexedPreference rating : snapshot.getRatings()) {
			double w = getWeight(rating.getItemId());
			AVector item = itemFeatures.getRow(rating.getItemIndex());
			AVector user = userFeatures.getRow(rating.getUserIndex());
			double prediction = item.dotProduct(user);
			double error = (rating.getValue() - prediction) * w;
			sum += Math.abs(error);
		}
		System.out.println("MAE " + sum / snapshot.getRatings().size());
	}
}
