package util;

import it.unimi.dsi.fastutil.longs.LongCollection;
import it.unimi.dsi.fastutil.longs.LongSet;
import org.grouplens.lenskit.data.event.Event;
import org.grouplens.lenskit.data.history.UserHistory;
import org.grouplens.lenskit.data.pref.IndexedPreference;
import org.grouplens.lenskit.data.snapshot.PreferenceSnapshot;
import org.grouplens.lenskit.knn.item.model.ItemItemModel;
import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.grouplens.lenskit.vectors.SparseVector;

import java.io.BufferedReader;
import java.util.*;

public class ContentUtil {
	private static int[] termDocFreq;

	public static double getJaccard(SparseVector vector1, SparseVector vector2) {
		if (vector1 == null || vector2 == null) {
			return 0.0;
		}
		int intersection = 0;
		for (long item : vector1.keySet()) {
			if (vector2.containsKey(item)) {
				intersection++;
			}
		}
		if (intersection == 0) {
			return 0.0;
		}
		int union = vector1.size() + vector2.size() - intersection;
		return (double) intersection / (double) union;
	}

	public static double getCos(SparseVector vector1, SparseVector vector2) {
		if (vector1 == null || vector2 == null) {
			return 0.0;
		}
		double dot = vector1.dot(vector2);
		double denom = vector1.norm() * vector2.norm();
		if (denom == 0) {
			return 0.0;
		}
		return dot / denom;
	}

	public static double getSim(SparseVector vector1, SparseVector vector2) {
		return getJaccard(vector1, vector2);
	}

	public static SparseVector getUserSparseVector(UserHistory<Event> events, Map<Long, SparseVector> itemContentMap) {
		LongSet set = events.itemSet();
		return getUserSparseVector(set, itemContentMap);
	}

	public static SparseVector getUserSparseVector(Collection<Long> itemIds, Map<Long, SparseVector> itemContentMap) {
		Map<Long, Double> prefMap = new HashMap<Long, Double>();
		for (Long itemId : itemIds) {
			if (!itemContentMap.containsKey(itemId)) {
				continue;
			}
			SparseVector vector = itemContentMap.get(itemId);
			for (long key : vector.keySet()) {
				double feature = 1.0;//vector.get(key);
				Double val = 0.0;
				if (prefMap.containsKey(key)) {
					val = prefMap.get(key);
				}
				val += feature;
				prefMap.put(key, val);
			}
		}
		return mapToVector(prefMap);
	}

	private static SparseVector mapToVector(Map<Long, Double> prefMap) {
		MutableSparseVector vector = MutableSparseVector.create(prefMap.keySet());
		for (Map.Entry<Long, Double> entry : prefMap.entrySet()) {
			vector.set(entry.getKey(), entry.getValue());
		}
		return vector;
	}

	public static Map<Long, SparseVector> getItemContentMap(String path) {
		Map<Long, SparseVector> itemContentMap = new HashMap<Long, SparseVector>();
		Map<Long, BitSet> vecMap = getVectors(path);
		for (Map.Entry<Long, BitSet> entry : vecMap.entrySet()) {
			SparseVector vector = vecByBitSet(entry.getValue(), vecMap.size());
			itemContentMap.put(entry.getKey(), vector);
		}
		return itemContentMap;
	}

	private static SparseVector vecByBitSet(BitSet set, int docNum) {
		Set<Long> keys = new HashSet<Long>();
		for (int i = 0; i < termDocFreq.length; i++) {
			keys.add((long) i);
		}
		MutableSparseVector vector = MutableSparseVector.create(keys);
		for (int i = 0; i < termDocFreq.length; i++) {
			if (set.get(i)) {
				double tfidf = 1.0;// getTFIDF(docNum, i);
				vector.set((long) i, tfidf);
			}
		}
		return vector;
	}

	private static double getTFIDF(int docNum, int termNumber) {
		double tf = 1.0 / termDocFreq.length;
		double idf = Math.log((double) docNum / termDocFreq[termNumber]);
		return tf * idf;
	}

	private static Map<Long, BitSet> getVectors(String path) {
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
}
