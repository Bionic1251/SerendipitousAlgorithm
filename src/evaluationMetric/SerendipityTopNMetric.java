package evaluationMetric;

import it.unimi.dsi.fastutil.longs.LongSet;
import org.grouplens.lenskit.Recommender;
import org.grouplens.lenskit.data.dao.UserEventDAO;
import org.grouplens.lenskit.data.dao.packed.RatingSnapshotDAO;
import org.grouplens.lenskit.data.event.Event;
import org.grouplens.lenskit.data.history.RatingVectorUserHistorySummarizer;
import org.grouplens.lenskit.data.history.UserHistory;
import org.grouplens.lenskit.data.source.DataSource;
import org.grouplens.lenskit.eval.Attributed;
import org.grouplens.lenskit.eval.data.traintest.TTDataSet;
import org.grouplens.lenskit.eval.metrics.topn.ItemSelector;
import org.grouplens.lenskit.knn.item.model.ItemItemBuildContext;
import org.grouplens.lenskit.knn.item.model.ItemItemBuildContextProvider;
import org.grouplens.lenskit.transform.normalize.DefaultUserVectorNormalizer;
import org.grouplens.lenskit.util.statistics.MeanAccumulator;
import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.grouplens.lenskit.vectors.SparseVector;

import java.util.*;

public class SerendipityTopNMetric extends PopSerendipityTopNMetric {
	private final Map<Long, SparseVector> itemContentMap;
	private final int longTailStart;

	public SerendipityTopNMetric(String suffix, int listSize, int number, ItemSelector candidates, ItemSelector exclude,
								 ItemSelector goodItems, Map<Long, SparseVector> itemContentMap, int longTailStart) {
		super(suffix, listSize, number, candidates, exclude, goodItems);
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
		Set<Long> midItems = getItemIds(expectedItemsNumber, longTailStart);
		addPersonalizedItems(midItems, uDao, expectedItemsNumber);
		Set<Long> longTailItems = getItemIds(longTailStart, expectedItemContainers.size());
		addPersonalizedItems(longTailItems, uDao, expectedItemsNumber / 2);
	}

	private void addPersonalizedItems(Set<Long> items, UserEventDAO uDao, int itemNumber) {
		for (Long userId : expectedMap.keySet()) {
			UserHistory<Event> userEvents = uDao.getEventsForUser(userId);
			SparseVector userVector = getUserSparseVector(userEvents);
			List<Long> expectedItems = getClosestItems(userVector, items, itemNumber);
			Set<Long> set = expectedMap.get(userId);
			set.addAll(expectedItems);
		}
	}

	private List<Long> getClosestItems(SparseVector userVec, Set<Long> items, int itemNumber){
		List<evaluationMetric.Container<Double>> containerList = new ArrayList<evaluationMetric.Container<Double>>();
		for(Long itemId : items){
			double sim = getCosine(userVec, itemContentMap.get(itemId));
			containerList.add(new evaluationMetric.Container<Double>(itemId, sim));
		}
		Collections.sort(containerList);
		Collections.reverse(containerList);
		List<Long> ids = new ArrayList<Long>();
		for(int i = 0; i < itemNumber; i++){
			ids.add(containerList.get(i).getId());
		}
		return ids;
	}

	private double getCosine(SparseVector vector1, SparseVector vector2) {
		double dot = vector1.dot(vector2);
		double denom = vector1.norm() * vector2.norm();
		if (denom == 0) {
			return 0.0;
		}
		return dot / denom;
	}

	private SparseVector getUserSparseVector(UserHistory<Event> events) {
		Map<Long, Double> prefMap = new HashMap<Long, Double>();
		LongSet set = events.itemSet();
		for (long itemId : set) {
			if (!itemContentMap.containsKey(itemId)) {
				continue;
			}
			SparseVector vector = itemContentMap.get(itemId);
			for (long key : vector.keySet()) {
				double feature = 1.0;//vector.get(key);
				Double val = 0.0;
				if (prefMap.containsKey(key)){
					val = prefMap.get(key);
				}
				val += feature;
				prefMap.put(key, val);
			}
		}
		return mapToVector(prefMap);
	}

	private SparseVector mapToVector(Map<Long, Double> prefMap){
		MutableSparseVector vector = MutableSparseVector.create(prefMap.keySet());
		for(Map.Entry<Long, Double> entry : prefMap.entrySet()){
			vector.set(entry.getKey(), entry.getValue());
		}
		return vector;
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
