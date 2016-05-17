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
import org.grouplens.lenskit.vectors.SparseVector;
import org.hamcrest.Matchers;
import util.PrepareUtil;
import util.Settings;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.*;


public class AggregateCustomSerendipityTopNMetric extends AbstractMetric<MeanAccumulator, AggregateCustomSerendipityTopNMetric.AggregateResult, AggregateCustomSerendipityTopNMetric.AggregateResult> {
	protected Map<Long, Set<Long>> expectedMap;
	protected List<Container<Double>> expectedItemContainers;
	private Set<Long> defaultExpectedItems;
	protected final int expectedItemsNumber;
	private final ItemSelector goodItems;
	private final ItemSelector candidates;
	private final ItemSelector exclude;
	private final String suffix;
	private String dataSetName;
	//private Map<Long, Double> ratingThresholds;

	private MeanAccumulator context1;
	private MeanAccumulator context5;
	private MeanAccumulator context10;
	private MeanAccumulator context15;
	private MeanAccumulator context20;
	private MeanAccumulator context25;
	private MeanAccumulator context30;

	public AggregateCustomSerendipityTopNMetric(String suffix, int number, ItemSelector candidates, ItemSelector exclude, ItemSelector goodItems) {
		super(AggregateResult.class, AggregateResult.class);
		this.goodItems = goodItems;
		this.suffix = suffix;
		expectedItemsNumber = number;
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
		double threshold = Settings.R_THRESHOLD;//getRaitingThreshold(user.getUserId());
		double ser1 = measureUser(user, context1, recommendations, 1, threshold);
		double ser5 = measureUser(user, context5, recommendations, 5, threshold);
		double ser10 = measureUser(user, context10, recommendations, 10, threshold);
		double ser15 = measureUser(user, context15, recommendations, 15, threshold);
		double ser20 = measureUser(user, context20, recommendations, 20, threshold);
		double ser25 = measureUser(user, context25, recommendations, 25, threshold);
		double ser30 = measureUser(user, context30, recommendations, 30, threshold);
		return new AggregateResult(ser1, ser5, ser10, ser15, ser20, ser25, ser30, 1);
	}

	protected double measureUser(TestUser user, MeanAccumulator context, List<ScoredId> recommendations, int listSize, double threshold) {
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
			if (goodItems.contains(key) && !getExpectedItems(user.getUserId()).contains(key)) {
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
		dataSetName = dataSet.getTrainingData().getName();
		//getRatingThresholds(dataSet);
		return new MeanAccumulator();
	}

	private double getRaitingThreshold(long userId) {
		Map<Long, Double> ratingThresholds = PrepareUtil.getAverageRatingMap(dataSetName);
		if (ratingThresholds.containsKey(userId)) {
			return ratingThresholds.get(userId);
		}
		return 3.0;
	}

	private void getRatingThresholds(TTDataSet dataSet) {
		//ratingThresholds = PrepareUtil.getAverageRatingMap(dataSet.getTrainingData().getName());
	}

	private void updateExpectedItems(TTDataSet dataSet) {
		updatePopItems(dataSet);
		updatePersItems(dataSet);
	}

	private void updatePersItems(TTDataSet dataSet) {
		String[] fsplit = dataSet.getName().split("\\.");
		String fileName = fsplit[fsplit.length - 1];
		BufferedReader reader = null;
		try {
			reader = new BufferedReader(new FileReader(fileName));
			String line = reader.readLine();
			while (line != null && !line.isEmpty()) {
				String[] split = line.split("\t");
				Long userId = Long.valueOf(split[0]);
				Set<Long> set = new HashSet<Long>(defaultExpectedItems);
				for (int i = 1; i <= Settings.ADDITIONAL_OBVIOUS && i < split.length; i++) {
					set.add(Long.valueOf(split[i]));
				}
				expectedMap.put(userId, set);
				line = reader.readLine();
			}
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			if (reader != null) {
				try {
					reader.close();
				} catch (Exception ex) {
					ex.printStackTrace();
				}
			}
		}
	}

	private void updatePopItems(TTDataSet dataSet) {
		defaultExpectedItems = new HashSet<Long>();
		expectedMap = new HashMap<Long, Set<Long>>();
		Map<Long, Double> popMap = PrepareUtil.getNormalizedPopMap(dataSet.getTrainingData().getName(), ",");
		expectedItemContainers = new ArrayList<Container<Double>>();
		for (Map.Entry<Long, Double> entry : popMap.entrySet()) {
			expectedItemContainers.add(new Container<Double>(entry.getKey(), entry.getValue()));
		}
		Collections.sort(expectedItemContainers);
		Collections.reverse(expectedItemContainers);
		for (int i = 0; i < expectedItemsNumber; i++) {
			defaultExpectedItems.add(expectedItemContainers.get(i).getId());
		}
	}

	private Set<Long> getExpectedItems(Long userId) {
		if (expectedMap.containsKey(userId)) {
			return expectedMap.get(userId);
		}
		return defaultExpectedItems;
	}

	public static class AggregateResult {
		@ResultColumn("SerCust1")
		public final double ser1;

		@ResultColumn("SerCust5")
		public final double ser5;

		@ResultColumn("SerCust10")
		public final double ser10;

		@ResultColumn("SerCust15")
		public final double ser15;

		@ResultColumn("SerCust20")
		public final double ser20;

		@ResultColumn("SerCust25")
		public final double ser25;

		@ResultColumn("SerCust30")
		public final double ser301;

		@ResultColumn("SerCustCount")
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
