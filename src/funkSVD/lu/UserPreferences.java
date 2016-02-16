package funkSVD.lu;

import it.unimi.dsi.fastutil.longs.LongCollection;
import org.grouplens.lenskit.data.pref.IndexedPreference;
import org.grouplens.lenskit.data.snapshot.PreferenceSnapshot;

import java.util.*;

public class UserPreferences {
	private final Map<Long, UserInfo> userInfoMap;

	public UserPreferences(PreferenceSnapshot snapshot, double threshold) {
		userInfoMap = new HashMap<Long, UserInfo>();
		LongCollection userIds = snapshot.getUserIds();
		for (long usedId : userIds) {
			Collection<IndexedPreference> userRatings = snapshot.getUserRatings(usedId);
			List<IndexedPreference> liked = new ArrayList<IndexedPreference>();
			List<IndexedPreference> disLiked = new ArrayList<IndexedPreference>();
			for (IndexedPreference pref : userRatings) {
				if (pref.getValue() <= threshold) {
					disLiked.add(pref);
				} else {
					liked.add(pref);
				}
			}
			userInfoMap.put(usedId, new UserInfo(disLiked, liked));
		}
	}

	public List<IndexedPreference> getDislikedItems(Long userId){
		return userInfoMap.get(userId).disliked;
	}

	public List<IndexedPreference> getLikedItems(Long userId){
		return userInfoMap.get(userId).liked;
	}

	private class UserInfo {
		private final List<IndexedPreference> liked;
		private final List<IndexedPreference> disliked;

		private UserInfo(List<IndexedPreference> disliked, List<IndexedPreference> liked) {
			this.disliked = disliked;
			this.liked = liked;
		}
	}
}
