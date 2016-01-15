package evaluationMetric;

import it.unimi.dsi.fastutil.longs.LongSortedSet;
import org.grouplens.lenskit.Recommender;
import org.grouplens.lenskit.eval.Attributed;
import org.grouplens.lenskit.eval.data.traintest.TTDataSet;
import org.grouplens.lenskit.eval.metrics.AbstractMetric;
import org.grouplens.lenskit.eval.metrics.ResultColumn;
import org.grouplens.lenskit.eval.traintest.TestUser;
import org.grouplens.lenskit.knn.item.model.ItemItemBuildContext;
import org.grouplens.lenskit.util.statistics.MeanAccumulator;
import org.grouplens.lenskit.vectors.SparseVector;

import java.util.*;

public class SMetric extends AbstractMetric<MeanAccumulator, SMetric.Result, SMetric.Result> {
	private Map<Long, Set<Long>> expectedMap;
	private Set<Long> defaultExpectedItems;
	private final int LIST_LENGTH = 2;

	public SMetric(ItemItemBuildContext context) {
		super(Result.class, Result.class);
		updateExpectedMap(context);
	}

	@Override
	protected Result doMeasureUser(TestUser user, MeanAccumulator context) {
		SparseVector predictions = user.getPredictions();
		if (predictions == null) {
			return null;
		}

		SparseVector ratings = user.getTestRatings();

		context.add(2.0);
		return new Result(1.0);
	}

	@Override
	protected Result getTypedResults(MeanAccumulator context) {
		return new Result(context.getMean());
	}

	@Override
	public MeanAccumulator createContext(Attributed algorithm, TTDataSet dataSet, Recommender recommender) {
		return new MeanAccumulator();
	}

	public void updateExpectedMap(ItemItemBuildContext dataContext) {
		defaultExpectedItems = new HashSet<Long>();
		expectedMap = new HashMap<Long, Set<Long>>();
		Set<Long> expectedItems = new HashSet<Long>();
		Set<Long> userSet = new HashSet<Long>();
		List<Container> expectedItemContaners = new ArrayList<Container>();
		LongSortedSet itemSet = dataContext.getItems();
		SparseVector itemVector;
		for (Long itemId : itemSet) {
			itemVector = dataContext.itemVector(itemId);
			expectedItemContaners.add(new Container(itemId, itemVector.size()));
			userSet.addAll(itemVector.keySet());
		}
		Collections.sort(expectedItemContaners);
		for (int i = 0; i < LIST_LENGTH; i++) {
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

	public static class Result {
		@ResultColumn("Serendipity")
		private final double value;

		public Result(double value) {
			this.value = value;
		}
	}
}
