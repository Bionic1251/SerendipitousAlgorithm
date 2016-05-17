package evaluationMetric;

import org.grouplens.lenskit.Recommender;
import org.grouplens.lenskit.data.dao.UserEventDAO;
import org.grouplens.lenskit.data.event.Event;
import org.grouplens.lenskit.data.history.UserHistory;
import org.grouplens.lenskit.data.source.DataSource;
import org.grouplens.lenskit.eval.Attributed;
import org.grouplens.lenskit.eval.data.traintest.TTDataSet;
import org.grouplens.lenskit.eval.metrics.topn.ItemSelector;
import org.grouplens.lenskit.util.statistics.MeanAccumulator;
import org.grouplens.lenskit.vectors.SparseVector;
import util.ContentUtil;

import java.util.*;

public class AggregateSerendipityTopNMetric extends AggregatePopSerendipityTopNMetric {
	private final Map<Long, SparseVector> itemContentMap;
	private final int longTailStart;

	public AggregateSerendipityTopNMetric(String suffix, int number, ItemSelector candidates, ItemSelector exclude,
										  ItemSelector goodItems, Map<Long, SparseVector> itemContentMap, int longTailStart) {
		super(suffix, number, candidates, exclude, goodItems);
		this.itemContentMap = itemContentMap;
		this.longTailStart = longTailStart;
	}

	@Override
	public MeanAccumulator createContext(Attributed algorithm, TTDataSet dataSet, Recommender recommender) {
		MeanAccumulator accumulator = super.createContext(algorithm, dataSet, recommender);
		DataSource source = dataSet.getTrainingData();
		UserEventDAO uDao = source.getUserEventDAO();
		updatePersonalizedExpectedItems(uDao);
		return accumulator;
	}

	private void updatePersonalizedExpectedItems(UserEventDAO uDao) {
		System.out.println("Collect expected items for serendipity metric");
		Set<Long> midItems = getItemIds(expectedItemsNumber, longTailStart);
		addPersonalizedItems(midItems, uDao, expectedItemsNumber * 3);
		/*System.out.println("Items from the mid part collected");
		Set<Long> longTailItems = getItemIds(longTailStart, expectedItemContainers.size());
		addPersonalizedItems(longTailItems, uDao, expectedItemsNumber);
		System.out.println("Items from the long tail collected");*/
	}

	private void addPersonalizedItems(Set<Long> items, UserEventDAO uDao, int itemNumber) {
		int i = 0;
		for (Long userId : expectedMap.keySet()) {
			i++;
			if (i % 100 == 0) {
				System.out.println(i + " users processed");
			}
			UserHistory<Event> userEvents = uDao.getEventsForUser(userId);
			Set<Long> ratedItems = new HashSet<Long>();
			for (Event event : userEvents) {
				ratedItems.add(event.getItemId());
			}
			List<Long> expectedItems = getMostSimilarItems(ratedItems, items, itemNumber);
			Set<Long> set = expectedMap.get(userId);
			set.addAll(expectedItems);
		}
		/*for (Long userId : expectedMap.keySet()) {
			UserHistory<Event> userEvents = uDao.getEventsForUser(userId);
			SparseVector userVector = ContentUtil.getUserSparseVector(userEvents, itemContentMap);
			List<Long> expectedItems = getClosestItems(userVector, items, itemNumber);
			Set<Long> set = expectedMap.get(userId);
			set.addAll(expectedItems);
		}*/
	}

	private List<Long> getMostSimilarItems(Collection<Long> ratedItems, Set<Long> items, int itemNumber) {
		List<evaluationMetric.Container<Double>> containerList = new ArrayList<evaluationMetric.Container<Double>>();
		for (Long itemId : items) {
			double avgSim = 0.0;
			for (Long ratedItemId : ratedItems) {
				avgSim += ContentUtil.getJaccard(itemContentMap.get(ratedItemId), itemContentMap.get(itemId));
			}
			containerList.add(new evaluationMetric.Container<Double>(itemId, avgSim / ratedItems.size()));
		}
		Collections.sort(containerList);
		Collections.reverse(containerList);
		List<Long> ids = new ArrayList<Long>();
		for (int i = 0; i < itemNumber; i++) {
			ids.add(containerList.get(i).getId());
		}
		return ids;
	}

	private List<Long> getClosestItems(SparseVector userVec, Set<Long> items, int itemNumber) {
		List<evaluationMetric.Container<Double>> containerList = new ArrayList<evaluationMetric.Container<Double>>();
		for (Long itemId : items) {
			double sim = ContentUtil.getJaccard(userVec, itemContentMap.get(itemId));
			containerList.add(new evaluationMetric.Container<Double>(itemId, sim));
		}
		Collections.sort(containerList);
		Collections.reverse(containerList);
		List<Long> ids = new ArrayList<Long>();
		for (int i = 0; i < itemNumber; i++) {
			ids.add(containerList.get(i).getId());
		}
		return ids;
	}

	private Set<Long> getItemIds(int start, int number) {
		Set<Long> ids = new HashSet<Long>();
		for (int i = start; i < number; i++) {
			Container container = expectedItemContainers.get(i);
			ids.add(container.getId());
		}
		return ids;
	}
}
