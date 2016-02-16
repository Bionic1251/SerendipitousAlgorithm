package lc;


import org.grouplens.lenskit.core.Transient;
import org.grouplens.lenskit.data.pref.IndexedPreference;
import org.grouplens.lenskit.data.snapshot.PreferenceSnapshot;

import javax.annotation.Nonnull;
import javax.inject.Inject;
import javax.inject.Provider;
import java.util.HashMap;
import java.util.Map;

public class AlgPopModelBuilder implements Provider<AlgPopModel> {
	protected final PreferenceSnapshot snapshot;

	@Inject
	public AlgPopModelBuilder(@Transient @Nonnull PreferenceSnapshot snapshot) {
		this.snapshot = snapshot;
	}

	@Override
	public AlgPopModel get() {
		Map<Long, Container> itemMap = new HashMap<Long, Container>();
		for (IndexedPreference rating : snapshot.getRatings()) {
			Container container = new Container(rating.getItemId());
			if (itemMap.containsKey(rating.getItemId())) {
				container = itemMap.get(rating.getItemId());
			}
			container.addRating(rating.getValue());
			itemMap.put(rating.getItemId(), container);
		}
		double maxNum = 0;
		for (Container container : itemMap.values()) {
			maxNum = Math.max(container.getRatingNumber(), maxNum);
		}

		for (Container container : itemMap.values()) {
			container.setRatingNumber(container.getRatingNumber() / maxNum);
		}
		return new AlgPopModel(itemMap);
	}

	public static class Container implements Comparable<Container> {
		private Long id;
		private Double ratingSum = 0.0;
		private Double ratingNumber = 0.0;

		private Container(Long id) {
			this.id = id;
		}

		public void addRating(double rating) {
			ratingSum += rating;
			ratingNumber++;
		}

		public void setRatingNumber(Double ratingNumber) {
			this.ratingNumber = ratingNumber;
		}

		public Double getRatingNumber() {
			return ratingNumber;
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
