package util;

import org.grouplens.lenskit.data.pref.IndexedPreference;
import org.grouplens.lenskit.data.snapshot.PreferenceSnapshot;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

public class Util {
	public static Map<Integer, Double> getPopMap(PreferenceSnapshot snapshot){
		Collection<IndexedPreference> ratings = snapshot.getRatings();
		Map<Integer, Double> popMap = new HashMap<Integer, Double>();
		double max = 0.0;
		for (IndexedPreference pref : ratings) {
			double val = 0.0;
			if (popMap.containsKey(pref.getItemIndex())) {
				val = popMap.get(pref.getItemIndex());
			}
			val++;
			if (val > max) {
				max = val;
			}
			popMap.put(pref.getItemIndex(), val);
		}
		for (Integer key : popMap.keySet()) {
			double val = popMap.get(key);
			val /= max;
			popMap.put(key, val);
		}
		return popMap;
	}
}
