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


public class SerendipityTopNMetric extends AbstractMetric<MeanAccumulator, SerendipityTopNMetric.Result, SerendipityTopNMetric.Result> {
	private Map<Long, Set<Long>> expectedMap;
	private Set<Long> defaultExpectedItems;
	private final int expectedItemsNumber;
	private final int evaluationListSize;
	private final ItemSelector goodItems;
	private final ItemSelector candidates;
	private final ItemSelector exclude;
	private final String suffix;

	public SerendipityTopNMetric(String suffix, int listSize, int number, ItemSelector candidates, ItemSelector exclude, ItemSelector goodItems) {
		super(SerendipityTopNMetric.Result.class, SerendipityTopNMetric.Result.class);
		this.evaluationListSize = listSize;
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
	protected SerendipityTopNMetric.Result doMeasureUser(TestUser user, MeanAccumulator context) {
		List<ScoredId> recommendations = user.getRecommendations(evaluationListSize, candidates, exclude);
		if (recommendations == null || recommendations.isEmpty()) {
			return null;
		}

		LongSet goodItems = this.goodItems.select(user);
		if (goodItems == null || goodItems.isEmpty()) {
			return new Result(0.0, 1);
		}

		Set<Long> serendipitousItems = new HashSet<Long>();
		SparseVector ratings = user.getTestRatings();
		for (Long key : ratings.keySet()) {
			if (goodItems.contains(key) && !getExpectedItems(user.getUserId()).contains(key)) {
				serendipitousItems.add(key);
			}
		}
		if (serendipitousItems.isEmpty()) {
			return new SerendipityTopNMetric.Result(0.0, 1);
		}
		int serendipityCount = 0;
		for (ScoredId scoredId : recommendations) {
			if (serendipitousItems.contains(scoredId.getId())) {
				serendipityCount++;
			}
		}
		double number = serendipitousItems.size() > evaluationListSize ? evaluationListSize : serendipitousItems.size();
		double value = (double) serendipityCount / number;
		context.add(value);
		return new SerendipityTopNMetric.Result(value, 1);
	}

	@Override
	protected SerendipityTopNMetric.Result getTypedResults(MeanAccumulator context) {
		return new SerendipityTopNMetric.Result(context.getMean(), context.getCount());
	}

	@Override
	public MeanAccumulator createContext(Attributed algorithm, TTDataSet dataSet, Recommender recommender) {
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
		List<Container> expectedItemContaners = new ArrayList<Container>();
		LongSortedSet itemSet = dataContext.getItems();
		SparseVector itemVector;
		for (Long itemId : itemSet) {
			itemVector = dataContext.itemVector(itemId);
			expectedItemContaners.add(new Container(itemId, itemVector.values().size()));
			userSet.addAll(itemVector.keySet());
		}
		Collections.sort(expectedItemContaners);
		Collections.reverse(expectedItemContaners);
		for (int i = 0; i < expectedItemsNumber; i++) {
			expectedItems.add(expectedItemContaners.get(i).id);
		}
		defaultExpectedItems = expectedItems;
		for (Long userId : userSet) {
			expectedMap.put(userId, expectedItems);
		}
	}

	private Set<Long> getExpectedItems(Long userId) {
		if (expectedMap.containsKey(userId)) {
			return expectedMap.get(userId);
		}
		return defaultExpectedItems;
	}

	public static class Result {
		@ResultColumn("Serendipity")
		public final double utility;

		@ResultColumn("SerendipityCount")
		public final long count;

		public Result(double util, long count) {
			utility = util;
			this.count = count;
		}
	}

	private class Container implements Comparable<Container> {
		private Long id;
		private Integer ratingNumber;

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
