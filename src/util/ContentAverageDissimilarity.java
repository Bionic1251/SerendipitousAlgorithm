package util;

import it.unimi.dsi.fastutil.longs.LongCollection;
import org.grouplens.lenskit.data.pref.IndexedPreference;
import org.grouplens.lenskit.data.snapshot.PreferenceSnapshot;
import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.grouplens.lenskit.vectors.SparseVector;

import java.io.BufferedReader;
import java.util.*;

public class ContentAverageDissimilarity {
	private int[] termDocFreq;
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

	private Map<Long, SparseVector> getItemContentMap(String path) {
		Map<Long, SparseVector> itemContentMap = new HashMap<Long, SparseVector>();
		Map<Long, BitSet> vecMap = getVectors(path);
		for (Map.Entry<Long, BitSet> entry : vecMap.entrySet()) {
			SparseVector vector = vecByBitSet(entry.getValue());
			itemContentMap.put(entry.getKey(), vector);
		}
		return itemContentMap;
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
				while (line != null) {
					String[] vector = line.split(",");
					if (termDocFreq == null) {
						len = vector.length - 1;
						termDocFreq = new int[len];
					}
					Long id = Long.valueOf(vector[0]);
					BitSet bitSet = new BitSet(len);
					for (int i = 1; i < vector.length - 1; i++) {
						if (Integer.valueOf(vector[i]) != 0) {
							bitSet.set(i);
							termDocFreq[i]++;
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
					SparseVector ratedItemVector = itemContentMap.get(rating.getItemId());
					dissimilaritySum += 1 - ContentUtil.getCosine(itemVector, ratedItemVector);
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
}
