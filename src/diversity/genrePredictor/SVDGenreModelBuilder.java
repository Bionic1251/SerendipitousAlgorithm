package diversity.genrePredictor;

import mikera.matrixx.Matrix;
import mikera.vectorz.AVector;
import mikera.vectorz.Vector;
import org.grouplens.lenskit.core.Transient;
import org.grouplens.lenskit.data.pref.IndexedPreference;
import org.grouplens.lenskit.data.snapshot.PreferenceSnapshot;
import org.grouplens.lenskit.iterative.StoppingCondition;
import org.grouplens.lenskit.iterative.TrainingLoopController;
import org.grouplens.lenskit.mf.funksvd.FeatureInfo;
import org.grouplens.lenskit.util.statistics.MeanAccumulator;
import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.grouplens.lenskit.vectors.SparseVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import util.ContentAverageDissimilarity;

import javax.annotation.Nonnull;
import javax.inject.Inject;
import javax.inject.Provider;
import java.util.*;

public class SVDGenreModelBuilder implements Provider<SVDGenreModel> {
	private static Logger logger = LoggerFactory.getLogger(SVDGenreModelBuilder.class);

	protected final int featureCount = 10;
	protected final double learningRate = 0.0001;
	protected final double regularization = 0.1;
	private final StoppingCondition stoppingCondition;
	protected final PreferenceSnapshot snapshot;
	private Map<Long, MutableSparseVector> userMap = new HashMap<Long, MutableSparseVector>();

	@Inject
	public SVDGenreModelBuilder(@Transient @Nonnull PreferenceSnapshot snapshot, StoppingCondition stop) {
		this.snapshot = snapshot;
		stoppingCondition = stop;
		fillUserMapRatings();
		//fillUserMapFreq();
	}

	private void fillUserMapRatings() {
		ContentAverageDissimilarity dissimilarity = ContentAverageDissimilarity.getInstance();
		Map<Long, Map<Long, MeanAccumulator>> meanUserMap = new HashMap<Long, Map<Long, MeanAccumulator>>();
		for (IndexedPreference preference : snapshot.getRatings()) {
			Long userId = preference.getUserId();
			Long itemId = preference.getItemId();
			Map<Long, MeanAccumulator> map = new HashMap<Long, MeanAccumulator>();
			if (meanUserMap.containsKey(userId)) {
				map = meanUserMap.get(userId);
			}
			SparseVector itemVector = dissimilarity.getItemContentMap().get(itemId);

			for (Long genre : itemVector.keySet()) {
				MeanAccumulator accumulator = new MeanAccumulator();
				if (map.containsKey(genre)) {
					accumulator = map.get(genre);
				}
				accumulator.add(preference.getValue());
				map.put(genre, accumulator);
			}
			meanUserMap.put(userId, map);
		}
		MutableSparseVector emptyVector = dissimilarity.getEmptyVector();
		for (Map.Entry<Long, Map<Long, MeanAccumulator>> meanUserEntry : meanUserMap.entrySet()) {
			MutableSparseVector userVector = emptyVector.mutableCopy();
			for (Map.Entry<Long, MeanAccumulator> genreEntry : meanUserEntry.getValue().entrySet()) {
				userVector.set(genreEntry.getKey(), genreEntry.getValue().getMean());
			}
			userMap.put(meanUserEntry.getKey(), userVector);
		}
	}

	private void fillUserMapFreq() {
		ContentAverageDissimilarity dissimilarity = ContentAverageDissimilarity.getInstance();
		MutableSparseVector emptyVector = dissimilarity.getEmptyVector();
		for (IndexedPreference preference : snapshot.getRatings()) {
			Long userId = preference.getUserId();
			Long itemId = preference.getItemId();
			MutableSparseVector userVector = emptyVector.mutableCopy();
			if (userMap.containsKey(userId)) {
				userVector = userMap.get(userId);
			}
			SparseVector itemVector = dissimilarity.getItemContentMap().get(itemId);
			userVector.add(itemVector);
			userMap.put(userId, userVector);
		}
	}

	@Override
	public SVDGenreModel get() {
		System.out.println(SVDGenreModelBuilder.class);
		int userCount = snapshot.getUserIds().size();
		Matrix userFeatures = Matrix.create(userCount, featureCount);

		ContentAverageDissimilarity dissimilarity = ContentAverageDissimilarity.getInstance();
		int genreCount = dissimilarity.getEmptyVector().keySet().size();
		Matrix itemFeatures = Matrix.create(genreCount, featureCount);

		logger.info("Building genrePredictor with {} features for {} ratings",
				featureCount, snapshot.getRatings().size());

		List<FeatureInfo> featureInfo = new ArrayList<FeatureInfo>(featureCount);

		Vector uvec = Vector.createLength(userCount);
		Vector ivec = Vector.createLength(genreCount);

		Random random = new Random();
		random.setSeed(123);

		for (int f = 0; f < featureCount; f++) {
			for (int i = 0; i < userCount; i++) {
				uvec.set(i, random.nextDouble());
			}
			for (int i = 0; i < genreCount; i++) {
				ivec.set(i, random.nextDouble());
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

		return new SVDGenreModel(userFeatures, itemFeatures, snapshot);
	}

	private void trainFeatures(Matrix userFeatures, Matrix itemFeatures) {
		ContentAverageDissimilarity dissimilarity = ContentAverageDissimilarity.getInstance();
		Set<Long> genres = dissimilarity.getEmptyVector().keySet();
		for (Map.Entry<Long, MutableSparseVector> userEntry : userMap.entrySet()) {
			int userIndex = snapshot.userIndex().getIndex(userEntry.getKey());
			for (long genre : genres) {
				int genreIndex = (int) genre;
				AVector item = itemFeatures.getRow(genreIndex);
				AVector user = userFeatures.getRow(userIndex);
				double prediction = item.dotProduct(user);
				double value = userEntry.getValue().get(genre);
				if (value == 0.0) {
					continue;
				}
				double error = (value - prediction);
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
	}

	private void calculateStatistics(Matrix userFeatures, Matrix itemFeatures) {
		double sum = 0;
		ContentAverageDissimilarity dissimilarity = ContentAverageDissimilarity.getInstance();
		Set<Long> genres = dissimilarity.getEmptyVector().keySet();
		for (Map.Entry<Long, MutableSparseVector> userEntry : userMap.entrySet()) {
			int userIndex = snapshot.userIndex().getIndex(userEntry.getKey());
			for (long genre : genres) {
				int genreIndex = (int) genre;
				AVector item = itemFeatures.getRow(genreIndex);
				AVector user = userFeatures.getRow(userIndex);
				double prediction = item.dotProduct(user);
				double value = userEntry.getValue().get(genre);
				if (value == 0.0) {
					continue;
				}
				double error = (value - prediction);
				sum += Math.abs(error);
			}
		}
		System.out.println("MAE " + sum / snapshot.getRatings().size());
	}
}
