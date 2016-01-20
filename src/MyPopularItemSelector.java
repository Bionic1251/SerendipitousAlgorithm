import com.google.common.base.Function;
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import it.unimi.dsi.fastutil.longs.LongSet;
import it.unimi.dsi.fastutil.longs.LongSets;
import org.grouplens.lenskit.Recommender;
import org.grouplens.lenskit.core.LenskitRecommender;
import org.grouplens.lenskit.cursors.Cursor;
import org.grouplens.lenskit.data.dao.ItemEventDAO;
import org.grouplens.lenskit.data.event.Event;
import org.grouplens.lenskit.data.history.ItemEventCollection;
import org.grouplens.lenskit.eval.metrics.topn.ItemSelector;
import org.grouplens.lenskit.eval.traintest.TestUser;
import org.grouplens.lenskit.util.ScoredItemAccumulator;
import org.grouplens.lenskit.util.TopNScoredItemAccumulator;

import javax.annotation.Nullable;

public class MyPopularItemSelector implements ItemSelector, Function<Recommender, LongSet> {
	private final LongSet set;
	private final LoadingCache<Recommender, LongSet> cache;

	public MyPopularItemSelector(LongSet set) {
		cache = CacheBuilder.newBuilder()
				.weakKeys()
				.build(CacheLoader.from(this));
		this.set = set;
	}

	@Override
	public LongSet select(TestUser user) {
		return cache.getUnchecked(user.getRecommender());
	}

	@Nullable
	@Override
	public LongSet apply(@Nullable Recommender input) {
		return set;
	}

	@Override
	public boolean equals(Object o) {
		if (this == o) return true;
		if (o == null || getClass() != o.getClass()) return false;

		MyPopularItemSelector that = (MyPopularItemSelector) o;

		if (set.equals(set)) return false;

		return true;
	}

	@Override
	public int hashCode() {
		return set.size();
	}

	/**
	 * Cache loader to extract the item universe from a recommender.
	 */
	private static class UniverseLoader extends CacheLoader<Recommender, LongSet> {
		private int count;

		private UniverseLoader(int count) {
			this.count = count;
		}

		public LongSet load(Recommender input) throws Exception {
			if (input == null) {
				return LongSets.EMPTY_SET;
			}
			LenskitRecommender rec = (LenskitRecommender) input;
			ItemEventDAO idao = rec.get(ItemEventDAO.class);
			ScoredItemAccumulator accum = new TopNScoredItemAccumulator(count);
			Cursor<ItemEventCollection<Event>> items = idao.streamEventsByItem();
			try {
				for (ItemEventCollection<Event> item : items) {
					accum.put(item.getItemId(), item.size());
				}
			} finally {
				items.close();
			}
			return accum.finishSet();
		}
	}
}