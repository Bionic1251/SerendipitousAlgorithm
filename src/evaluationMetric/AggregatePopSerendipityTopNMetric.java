package evaluationMetric;

import it.unimi.dsi.fastutil.longs.LongSet;
import it.unimi.dsi.fastutil.longs.LongSortedSet;
import org.grouplens.lenskit.Recommender;
import org.grouplens.lenskit.data.dao.packed.RatingSnapshotDAO;
import org.grouplens.lenskit.data.history.RatingVectorUserHistorySummarizer;
import org.grouplens.lenskit.eval.Attributed;
import org.grouplens.lenskit.eval.data.traintest.TTDataSet;
import org.grouplens.lenskit.eval.metrics.AbstractMetric;
import org.grouplens.lenskit.eval.metrics.ResultColumn;
import org.grouplens.lenskit.eval.metrics.topn.ItemSelector;
import org.grouplens.lenskit.eval.traintest.TestUser;
import org.grouplens.lenskit.knn.item.model.ItemItemBuildContext;
import org.grouplens.lenskit.knn.item.model.ItemItemBuildContextProvider;
import org.grouplens.lenskit.scored.ScoredId;
import org.grouplens.lenskit.transform.normalize.DefaultUserVectorNormalizer;
import org.grouplens.lenskit.util.statistics.MeanAccumulator;
import org.grouplens.lenskit.vectors.SparseVector;

import java.util.*;


public class AggregatePopSerendipityTopNMetric extends AbstractMetric<MeanAccumulator, AggregatePopSerendipityTopNMetric.AggregateResult, AggregatePopSerendipityTopNMetric.AggregateResult> {
	protected Map<Long, Set<Long>> expectedMap;
	protected List<Container> expectedItemContainers;
	private Set<Long> defaultExpectedItems;
	protected final int expectedItemsNumber;
	private final ItemSelector goodItems;
	private final ItemSelector candidates;
	private final ItemSelector exclude;
	private final String suffix;

	private MeanAccumulator context1;
	private MeanAccumulator context5;
	private MeanAccumulator context10;
	private MeanAccumulator context15;
	private MeanAccumulator context20;
	private MeanAccumulator context25;
	private MeanAccumulator context30;

	public AggregatePopSerendipityTopNMetric(String suffix, int number, ItemSelector candidates, ItemSelector exclude, ItemSelector goodItems) {
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
		double ser1 = measureUser(user, context1, recommendations, 1);
		double ser5 = measureUser(user, context5, recommendations, 5);
		double ser10 = measureUser(user, context10, recommendations, 10);
		double ser15 = measureUser(user, context15, recommendations, 15);
		double ser20 = measureUser(user, context20, recommendations, 20);
		double ser25 = measureUser(user, context25, recommendations, 25);
		double ser30 = measureUser(user, context30, recommendations, 30);
		return new AggregateResult(ser1, ser5, ser10, ser15, ser20, ser25, ser30, 1);
	}

	protected double measureUser(TestUser user, MeanAccumulator context, List<ScoredId> recommendations, int listSize) {
		if (recommendations.size() > listSize) {
			recommendations = new ArrayList<ScoredId>(recommendations.subList(0, listSize));
		}
		LongSet goodItems = this.goodItems.select(user);
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
		return new MeanAccumulator();
	}

	private void updateExpectedItems(TTDataSet dataSet) {
		RatingSnapshotDAO.Builder builder = new RatingSnapshotDAO.Builder(dataSet.getTrainingDAO(), false);
		ItemItemBuildContextProvider provider = new ItemItemBuildContextProvider(builder.get(), new DefaultUserVectorNormalizer(), new RatingVectorUserHistorySummarizer());
		updateExpectedMap(provider.get());
	}

	private void updateExpectedMap(ItemItemBuildContext dataContext) {
		defaultExpectedItems = new HashSet<Long>();
		expectedMap = new HashMap<Long, Set<Long>>();
		Set<Long> expectedItems = new HashSet<Long>();
		Set<Long> userSet = new HashSet<Long>();
		expectedItemContainers = new ArrayList<Container>();
		LongSortedSet itemSet = dataContext.getItems();
		SparseVector itemVector;
		for (Long itemId : itemSet) {
			itemVector = dataContext.itemVector(itemId);
			expectedItemContainers.add(new Container(itemId, itemVector.values().size()));
			userSet.addAll(itemVector.keySet());
		}
		Collections.sort(expectedItemContainers);
		Collections.reverse(expectedItemContainers);
		for (int i = 0; i < expectedItemsNumber; i++) {
			expectedItems.add(expectedItemContainers.get(i).id);
		}
		defaultExpectedItems = expectedItems;
		for (Long userId : userSet) {
			expectedMap.put(userId, new HashSet<Long>(expectedItems));
		}
	}

	private Set<Long> getExpectedItems(Long userId) {
		if (expectedMap.containsKey(userId)) {
			return expectedMap.get(userId);
		}
		return defaultExpectedItems;
	}

	public static class AggregateResult {
		@ResultColumn("Serendipity1")
		public final double ser1;

		@ResultColumn("Serendipity5")
		public final double ser5;

		@ResultColumn("Serendipity10")
		public final double ser10;

		@ResultColumn("Serendipity15")
		public final double ser15;

		@ResultColumn("Serendipity20")
		public final double ser20;

		@ResultColumn("Serendipity25")
		public final double ser25;

		@ResultColumn("Serendipity30")
		public final double ser301;

		@ResultColumn("SerendipityCount")
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

	protected class Container implements Comparable<Container> {
		private Long id;
		private Integer ratingNumber;

		public Long getId() {
			return id;
		}

		private Container(Long id, int ratingNumber) {
			this.id = id;
			this.ratingNumber = ratingNumber;
		}

		@Override
		public boolean equals(Object obj) {
			Container container = (Container) obj;
			return container.id.equals(id);
		}

		@Override
		public int compareTo(Container o) {
			return ratingNumber.compareTo(o.ratingNumber);
		}
	}
}
