package util;

import it.unimi.dsi.fastutil.longs.LongCollection;
import org.grouplens.lenskit.data.pref.IndexedPreference;
import org.grouplens.lenskit.data.snapshot.PreferenceSnapshot;
import org.grouplens.lenskit.knn.item.model.ItemItemModel;
import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.grouplens.lenskit.vectors.SparseVector;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

public class CollaborativeUtil {
	public static Map<Long, SparseVector> getUserItemMap(PreferenceSnapshot snapshot, ItemItemModel itemItemModel) {
		System.out.println("Collaborative distance calculation");
		Map<Long, SparseVector> userItemDissimilarityMap = new HashMap<Long, SparseVector>();
		LongCollection userIds = snapshot.getUserIds();
		LongCollection itemIds = snapshot.getItemIds();
		int size = userIds.size();
		int maxCount = 0;
		for (long userId : userIds) {
			MutableSparseVector itemDissimilarityVector = MutableSparseVector.create(itemIds, 0.0);
			Collection<IndexedPreference> ratings = snapshot.getUserRatings(userId);
			for (long itemId : itemIds) {
				double dissimilaritySum = 0;
				int count = 0;
				for (IndexedPreference rating : ratings) {
					SparseVector vector = itemItemModel.getNeighbors(itemId);
					if (rating.getItemId() == itemId) {
						continue;
					}
					if (!vector.containsKey(rating.getItemId())) {
						dissimilaritySum += 1;
					} else {
						dissimilaritySum += 1 - vector.get(rating.getItemId());
					}
					count++;
				}
				maxCount = Math.max(maxCount, count);
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
		System.out.println("Count " + maxCount);
		return userItemDissimilarityMap;
	}
}
