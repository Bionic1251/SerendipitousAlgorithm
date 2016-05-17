package evaluationMetric;

import it.unimi.dsi.fastutil.longs.LongSet;
import org.grouplens.lenskit.Recommender;
import org.grouplens.lenskit.eval.Attributed;
import org.grouplens.lenskit.eval.data.traintest.TTDataSet;
import org.grouplens.lenskit.eval.metrics.AbstractMetric;
import org.grouplens.lenskit.eval.metrics.ResultColumn;
import org.grouplens.lenskit.eval.metrics.topn.ItemSelector;
import org.grouplens.lenskit.eval.metrics.topn.ItemSelectors;
import org.grouplens.lenskit.eval.traintest.TestUser;
import org.grouplens.lenskit.scored.ScoredId;
import org.grouplens.lenskit.util.statistics.MeanAccumulator;
import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.grouplens.lenskit.vectors.SparseVector;
import org.hamcrest.Matchers;
import util.ContentAverageDissimilarity;
import util.PrepareUtil;
import util.Settings;
import util.Util;

import java.util.*;


public class AggregateGenresSerendipityTopNMetric extends AbstractMetric<MeanAccumulator, AggregateGenresSerendipityTopNMetric.AggregateResult, AggregateGenresSerendipityTopNMetric.AggregateResult> {
	private final ItemSelector candidates;
	private final ItemSelector exclude;
	private final String suffix;
	protected final int expectedItemsNumber;
	private Set<Long> defaultExpectedItems;

	private MeanAccumulator context1;
	private MeanAccumulator context5;
	private MeanAccumulator context10;
	private MeanAccumulator context15;
	private MeanAccumulator context20;
	private MeanAccumulator context25;
	private MeanAccumulator context30;

	public AggregateGenresSerendipityTopNMetric(String suffix, int number, ItemSelector candidates, ItemSelector exclude) {
		super(AggregateResult.class, AggregateResult.class);
		expectedItemsNumber = number;
		this.suffix = suffix;
		this.candidates = candidates;
		this.exclude = exclude;
	}

	@Override
	protected String getSuffix() {
		return suffix;
	}

	@Override
	protected AggregateResult doMeasureUser(TestUser user, MeanAccumulator context) {
		List<ScoredId> recommendations = user.getRecommendations(30, candidates, exclude);
		if (recommendations == null || recommendations.isEmpty()) {
			context1.add(0.0);
			context5.add(0.0);
			context10.add(0.0);
			context15.add(0.0);
			context20.add(0.0);
			context25.add(0.0);
			context30.add(0.0);
			return null;
		}
		Set<Long> genres = Util.getGenresByTestUser(user);
		double threshold = Settings.R_THRESHOLD;
		double ser1 = measureUser(user, context1, recommendations, 1, threshold, genres);
		double ser5 = measureUser(user, context5, recommendations, 5, threshold, genres);
		double ser10 = measureUser(user, context10, recommendations, 10, threshold, genres);
		double ser15 = measureUser(user, context15, recommendations, 15, threshold, genres);
		double ser20 = measureUser(user, context20, recommendations, 20, threshold, genres);
		double ser25 = measureUser(user, context25, recommendations, 25, threshold, genres);
		double ser30 = measureUser(user, context30, recommendations, 30, threshold, genres);
		return new AggregateResult(ser1, ser5, ser10, ser15, ser20, ser25, ser30, 1);
	}

	protected double measureUser(TestUser user, MeanAccumulator context, List<ScoredId> recommendations, int listSize, double threshold, Set<Long> genres) {
		if (recommendations.size() > listSize) {
			recommendations = new ArrayList<ScoredId>(recommendations.subList(0, listSize));
		}
		ItemSelector goodItemsSelector;
		goodItemsSelector = ItemSelectors.testRatingMatches(Matchers.greaterThan(threshold));
		LongSet goodItems = goodItemsSelector.select(user);
		if (goodItems == null || goodItems.isEmpty()) {
			return 0.0;
		}

		Set<Long> serendipitousItems = new HashSet<Long>();
		SparseVector ratings = user.getTestRatings();
		for (Long key : ratings.keySet()) {
			if (goodItems.contains(key) && !defaultExpectedItems.contains(key) && hasGenre(genres, key)) {
				serendipitousItems.add(key);
			}
		}
		if (serendipitousItems.isEmpty()) {
			return 0.0;
		}
		int serendipityCount = 0;
		for (ScoredId scoredId : recommendations) {
			if (serendipitousItems.contains(scoredId.getId())) {
				serendipityCount++;
			}
		}
		//double number = serendipitousItems.size() > evaluationListSize ? evaluationListSize : serendipitousItems.size();
		double value = (double) serendipityCount / listSize;
		context.add(value);
		return value;
	}

	private void updateExpectedItems(TTDataSet dataSet) {
		defaultExpectedItems = new HashSet<Long>();
		Set<Long> expectedItems = new HashSet<Long>();
		Map<Long, Double> popMap = PrepareUtil.getNormalizedPopMap(Settings.DATASET, "\t");
		List<Container<Double>> expectedItemContainers = new ArrayList<Container<Double>>();
		for (Map.Entry<Long, Double> entry : popMap.entrySet()) {
			expectedItemContainers.add(new Container<Double>(entry.getKey(), entry.getValue()));
		}
		Collections.sort(expectedItemContainers);
		Collections.reverse(expectedItemContainers);
		for (int i = 0; i < expectedItemsNumber; i++) {
			expectedItems.add(expectedItemContainers.get(i).getId());
		}
		defaultExpectedItems = expectedItems;
	}

	private boolean hasGenre(Set<Long> genres, long itemId) {
		ContentAverageDissimilarity dissimilarity = ContentAverageDissimilarity.getInstance();
		Map<Long, SparseVector> map = dissimilarity.getItemContentMap();
		if (!map.containsKey(itemId)) {
			return false;
		}
		SparseVector vector = map.get(itemId);
		for (long key : vector.keySet()) {
			if (vector.get(key) != 0 && genres.contains(key)) {
				return true;
			}
		}
		return false;
	}

	@Override
	protected AggregateResult getTypedResults(MeanAccumulator context) {
		long count = context30.getCount();
		double ser1 = context1.getMean();
		double ser5 = context5.getMean();
		double ser10 = context10.getMean();
		double ser15 = context15.getMean();
		double ser20 = context20.getMean();
		double ser25 = context25.getMean();
		double ser30 = context30.getMean();
		return new AggregateResult(ser1, ser5, ser10, ser15, ser20, ser25, ser30, count);
	}

	@Override
	public MeanAccumulator createContext(Attributed algorithm, TTDataSet dataSet, Recommender recommender) {
		context1 = new MeanAccumulator();
		context5 = new MeanAccumulator();
		context10 = new MeanAccumulator();
		context15 = new MeanAccumulator();
		context20 = new MeanAccumulator();
		context25 = new MeanAccumulator();
		context30 = new MeanAccumulator();
		updateExpectedItems(dataSet);
		return new MeanAccumulator();
	}


	public static class AggregateResult {
		@ResultColumn("SerendipityGenres1")
		public final double ser1;

		@ResultColumn("SerendipityGenres5")
		public final double ser5;

		@ResultColumn("SerendipityGenres10")
		public final double ser10;

		@ResultColumn("SerendipityGenres15")
		public final double ser15;

		@ResultColumn("SerendipityGenres20")
		public final double ser20;

		@ResultColumn("SerendipityGenres25")
		public final double ser25;

		@ResultColumn("SerendipityGenres30")
		public final double ser301;

		@ResultColumn("SerendipityGenresCount")
		public final long count;

		public AggregateResult(double ser1, double ser5, double ser10, double ser15, double ser20, double ser25, double ser301, long count) {
			this.ser1 = ser1;
			this.ser5 = ser5;
			this.ser10 = ser10;
			this.ser15 = ser15;
			this.ser20 = ser20;
			this.ser25 = ser25;
			this.ser301 = ser301;
			this.count = count;
		}
	}
}
