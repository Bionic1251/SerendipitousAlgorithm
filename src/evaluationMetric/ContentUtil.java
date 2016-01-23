package evaluationMetric;

import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.grouplens.lenskit.vectors.SparseVector;

import java.io.BufferedReader;
import java.util.*;

public class ContentUtil {
	private static int[] termDocFreq;

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
				double tfidf = getTFIDF(docNum, i);
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
		Map<Long, BitSet> vecMap = new HashMap<Long, BitSet>();
		int boolLen = 23, boolStart = 5;
		termDocFreq = new int[boolLen - boolStart + 1];
		try {
			BufferedReader reader = new BufferedReader(new java.io.FileReader(path));
			try {
				String line = reader.readLine();
				while (line != null) {
					String[] vector = line.split("\\|");
					Long id = Long.valueOf(vector[0]);
					BitSet bitSet = new BitSet(boolLen - boolStart + 1);
					for (int i = boolStart; i <= boolLen; i++) {
						if (Integer.valueOf(vector[i]) != 0) {
							bitSet.set(i - boolStart);
							termDocFreq[i - boolStart]++;
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
