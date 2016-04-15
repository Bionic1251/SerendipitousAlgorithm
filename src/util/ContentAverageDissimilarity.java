package util;

import it.unimi.dsi.fastutil.longs.LongCollection;
import lc.Normalizer;
import org.grouplens.lenskit.data.pref.IndexedPreference;
import org.grouplens.lenskit.data.snapshot.PreferenceSnapshot;
import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.grouplens.lenskit.vectors.SparseVector;
import org.grouplens.lenskit.vectors.VectorEntry;
import pop.PopModel;

import java.io.BufferedReader;
import java.util.*;

public class ContentAverageDissimilarity {
	private int[] termDocFreq;
	private String[] title;
	private final Map<Long, SparseVector> itemContentMap;
	private static ContentAverageDissimilarity instance;

	private ContentAverageDissimilarity(String path) {
		itemContentMap = getItemContentMap(path);
	}

	public static void create(String path) {
		instance = new ContentAverageDissimilarity(path);
	}

	public static ContentAverageDissimilarity getInstance() {
		if (instance == null) {
			System.out.println("ContentAverageDissimilarity does not exist");
		}
		return instance;
	}

	public Map<Long, SparseVector> getItemContentMap() {
		return itemContentMap;
	}

	public SparseVector toTFIDF(SparseVector inputVector) {
		MutableSparseVector vector = inputVector.mutableCopy();
		int sum = 0;
		for (int freq : termDocFreq) {
			sum += freq;
		}
		for (long key : inputVector.keySet()) {
			double val = inputVector.get(key);
			double idf = Math.log((double) sum / termDocFreq[(int) key]);
			vector.set(key, val * idf);
		}
		return vector;
	}

	private Map<Long, SparseVector> getItemContentMap(String path) {
		Map<Long, SparseVector> itemContentMap = new HashMap<Long, SparseVector>();
		Map<Long, BitSet> vecMap = getVectors(path);
		for (Map.Entry<Long, BitSet> entry : vecMap.entrySet()) {
			SparseVector vector = vecByBitSet(entry.getValue());
			itemContentMap.put(entry.getKey(), vector);
		}
		return itemContentMap;
	}

	public MutableSparseVector getEmptyVector() {
		int num = termDocFreq.length;
		Collection<Long> features = new ArrayList<Long>();
		for (long i = 0; i < num; i++) {
			features.add(i);
		}
		return MutableSparseVector.create(features, 0.0);
	}

	private SparseVector vecByBitSet(BitSet set) {
		Set<Long> keys = new HashSet<Long>();
		for (int i = 0; i < termDocFreq.length; i++) {
			keys.add((long) i);
		}
		MutableSparseVector vector = MutableSparseVector.create(keys);
		for (int i = 0; i < termDocFreq.length; i++) {
			if (set.get(i)) {
				double tfidf = 1.0;
				vector.set((long) i, tfidf);
			}
		}
		return vector;
	}

	private Map<Long, BitSet> getVectors(String path) {
		int len = 0;
		Map<Long, BitSet> vecMap = new HashMap<Long, BitSet>();
		termDocFreq = null;
		try {
			BufferedReader reader = new BufferedReader(new java.io.FileReader(path));
			try {
				String line = reader.readLine();
				title = line.split(",");
				if (title[0].equals("id")) {
					line = reader.readLine();
				}
				while (line != null) {
					String[] vector = line.split(",");
					if (termDocFreq == null) {
						len = vector.length - 1;
						termDocFreq = new int[len];
					}
					Long id = Long.valueOf(vector[0]);
					BitSet bitSet = new BitSet(len);
					for (int i = 1; i < vector.length; i++) {
						if (Integer.valueOf(vector[i]) != 0) {
							bitSet.set(i - 1);
							termDocFreq[i - 1]++;
						}
					}
					vecMap.put(id, bitSet);
					line = reader.readLine();
				}
			} finally {
				reader.close();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return vecMap;
	}

	public Map<Long, SparseVector> getUserItemAvgDistanceMap(PreferenceSnapshot snapshot) {
		System.out.println("Content distance calculation");
		Map<Long, SparseVector> userItemDissimilarityMap = new HashMap<Long, SparseVector>();
		LongCollection userIds = snapshot.getUserIds();
		LongCollection itemIds = snapshot.getItemIds();
		int size = userIds.size();
		for (long userId : userIds) {
			MutableSparseVector itemDissimilarityVector = MutableSparseVector.create(itemIds, 0.0);
			Collection<IndexedPreference> ratings = snapshot.getUserRatings(userId);
			for (long itemId : itemIds) {
				double dissimilaritySum = 0;
				int count = 0;
				SparseVector itemVector = itemContentMap.get(itemId);
				for (IndexedPreference rating : ratings) {
					if (rating.getItemId() == itemId) {
						continue;
					}
					SparseVector ratedItemVector = itemContentMap.get(rating.getItemId());
					dissimilaritySum += getDissimilarity(ratedItemVector, itemVector);
					count++;
				}
				if (count == 0) {
					continue;
				}
				dissimilaritySum = dissimilaritySum / count;
				if (Double.isNaN(dissimilaritySum)) {
					System.out.println(count + " !!!");
				}
				itemDissimilarityVector.set(itemId, dissimilaritySum);
			}
			userItemDissimilarityMap.put(userId, itemDissimilarityVector);
			size--;
			if (size % 100 == 0) {
				System.out.println(size + " left");
			}
		}
		return userItemDissimilarityMap;
	}

	private double getDissimilarity(SparseVector vec1, SparseVector vec2) {
		return 1 - ContentUtil.getSim(vec1, vec2);
	}

	public double getAverageDissimilarity(Long userId, Long itemId, PreferenceSnapshot snapshot) {
		Collection<IndexedPreference> ratings = snapshot.getUserRatings(userId);
		if (ratings == null || ratings.isEmpty()) {
			return 1;
		}
		double dissimilaritySum = 0;
		SparseVector itemVector = itemContentMap.get(itemId);
		for (IndexedPreference rating : ratings) {
			SparseVector ratedItemVector = itemContentMap.get(rating.getItemId());
			dissimilaritySum += getDissimilarity(ratedItemVector, itemVector);
		}
		dissimilaritySum = dissimilaritySum / ratings.size();
		if (Double.isNaN(dissimilaritySum)) {
			System.out.println(ratings.size() + " !!!");
		}
		return dissimilaritySum;
	}

	public SparseVector getAverageDissimilarity(Long userId, PreferenceSnapshot snapshot) {
		LongCollection itemIds = snapshot.getItemIds();
		MutableSparseVector itemDissimilarityVector = MutableSparseVector.create(itemIds, 1.0);
		Collection<IndexedPreference> ratings = snapshot.getUserRatings(userId);
		if (ratings == null || ratings.isEmpty()) {
			return itemDissimilarityVector;
		}
		for (Long itemId : itemIds) {
			double dissimilaritySum = 0;
			SparseVector itemVector = itemContentMap.get(itemId);
			for (IndexedPreference rating : ratings) {
				SparseVector ratedItemVector = itemContentMap.get(rating.getItemId());
				dissimilaritySum += getDissimilarity(ratedItemVector, itemVector);
			}
			dissimilaritySum = dissimilaritySum / ratings.size();
			if (Double.isNaN(dissimilaritySum)) {
				System.out.println(ratings.size() + " !!!");
			}
			itemDissimilarityVector.set(itemId, dissimilaritySum);
		}
		return itemDissimilarityVector;
	}

	public Map<Long, AverageAggregate> getAverageMap(PreferenceSnapshot snapshot, PopModel popModel, Map<Long, SparseVector> userItemDissimilarityMap) {
		System.out.println("Average Map Calculation");
		Map<Long, AverageAggregate> userThresholdMap = new HashMap<Long, AverageAggregate>();
		LongCollection userIdCollection = snapshot.getUserIds();
		int j = userIdCollection.size();
		for (long userId : userIdCollection) {
			j--;
			if (j % 100 == 0) {
				System.out.println(j + " users left");
			}
			UserStatistics tr = getAverageRating(userId, snapshot);
			UserStatistics td = getAverageProfileDissimilarity(userId, snapshot, userItemDissimilarityMap);
			UserStatistics tu = getAverageUnpopularity(userId, snapshot, popModel);
			userThresholdMap.put(userId, new AverageAggregate(tr, td, tu));
		}
		return userThresholdMap;
	}

	private UserStatistics getAverageRating(Long userId, PreferenceSnapshot snapshot) {
		Collection<IndexedPreference> preferences = snapshot.getUserRatings(userId);
		double min = Double.MAX_VALUE;
		double max = Double.MIN_VALUE;
		double rating = 0;
		if (preferences.isEmpty()) {
			return new UserStatistics(3, new Normalizer(3, 3));
		}
		for (IndexedPreference pref : preferences) {
			rating += pref.getValue();
			min = Math.min(min, pref.getValue());
			max = Math.max(max, pref.getValue());
		}
		rating /= preferences.size();
		return new UserStatistics(rating, new Normalizer(min, max));
	}

	private UserStatistics getAverageUnpopularity(Long userId, PreferenceSnapshot snapshot, PopModel popModel) {
		Collection<IndexedPreference> preferences = snapshot.getUserRatings(userId);
		double unpop = 0;
		double min = Double.MAX_VALUE;
		double max = Double.MIN_VALUE;
		Set<Long> set = prefsToSet(preferences);
		if (set.isEmpty()) {
			return new UserStatistics(1, new Normalizer(1, 1));
		}
		for (Long itemId : set) {
			double u = 1 - popModel.getPop(itemId) / popModel.getMax();
			unpop += u;
			min = Math.min(min, u);
			max = Math.max(max, u);
		}
		unpop /= set.size();
		return new UserStatistics(unpop, new Normalizer(min, max));
	}

	private UserStatistics getAverageProfileDissimilarity(Long userId, PreferenceSnapshot snapshot, Map<Long, SparseVector> userItemDissimilarityMap) {
		Collection<IndexedPreference> preferences = snapshot.getUserRatings(userId);
		double dissim = 0;
		double min = Double.MAX_VALUE;
		double max = Double.MIN_VALUE;
		Set<Long> set = prefsToSet(preferences);
		if (!userItemDissimilarityMap.containsKey(userId)) {
			return new UserStatistics(1, new Normalizer(1, 1));
		}
		SparseVector vec = userItemDissimilarityMap.get(userId);
		for (Long itemId : set) {
			dissim += vec.get(itemId);
			min = Math.min(min, vec.get(itemId));
			max = Math.max(max, vec.get(itemId));
		}
		dissim /= set.size();
		return new UserStatistics(dissim, new Normalizer(min, max));
	}

	private Set<Long> prefsToSet(Collection<IndexedPreference> preferences) {
		Set<Long> set = new HashSet<Long>();
		for (IndexedPreference pref : preferences) {
			set.add(pref.getItemId());
		}
		return set;
	}
}
