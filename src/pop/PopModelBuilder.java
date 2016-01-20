package pop;


import org.grouplens.lenskit.core.Transient;
import org.grouplens.lenskit.data.pref.IndexedPreference;
import org.grouplens.lenskit.data.snapshot.PreferenceSnapshot;

import javax.annotation.Nonnull;
import javax.inject.Inject;
import javax.inject.Provider;
import java.util.*;

public class PopModelBuilder implements Provider<PopModel> {
	protected final PreferenceSnapshot snapshot;

	@Inject
	public PopModelBuilder(@Transient @Nonnull PreferenceSnapshot snapshot) {
		this.snapshot = snapshot;
	}

	@Override
	public PopModel get() {
		Map<Long, Container> itemMap = new HashMap<Long, Container>();
		for (IndexedPreference rating : snapshot.getRatings()) {
			Container container = new Container(rating.getItemId());
			if (itemMap.containsKey(rating.getItemId())) {
				container = itemMap.get(rating.getItemId());
			}
			container.addRating(rating.getValue());
			itemMap.put(rating.getItemId(), container);
		}
		List<Container> containerList = new ArrayList<Container>(itemMap.values());
		Collections.sort(containerList);
		//Collections.reverse(containerList);

		List<Long> list = new ArrayList<Long>();
		for(int i = 0; i < containerList.size(); i++){
			list.add(containerList.get(i).getId());
		}
		return new PopModel(list);
	}

	private class Container implements Comparable<Container> {
		private Long id;
		private Double ratingSum = 0.0;
		private Integer ratingNumber = 0;

		private Container(Long id) {
			this.id = id;
		}

		public void addRating(double rating) {
			ratingSum += rating;
			ratingNumber++;
		}

		public Long getId() {
			return id;
		}

		@Override
		public int compareTo(Container o) {
			return ratingNumber.compareTo(o.ratingNumber);

			/*Double averageRating1 = ratingSum / ratingNumber;
			Double averageRating2 = o.ratingSum / o.ratingNumber;

			int res = averageRating1.compareTo(averageRating2);
			if (res == 0) {
				return ratingNumber.compareTo(o.ratingNumber);
			}
			return res;*/
		}
	}
}
