package content;

import annotation.TFIDF;
import it.unimi.dsi.fastutil.longs.LongCollection;
import org.grouplens.lenskit.basic.AbstractItemScorer;
import org.grouplens.lenskit.core.Transient;
import org.grouplens.lenskit.data.pref.IndexedPreference;
import org.grouplens.lenskit.data.snapshot.PreferenceSnapshot;
import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.grouplens.lenskit.vectors.SparseVector;
import org.grouplens.lenskit.vectors.VectorEntry;
import pop.PopModel;
import util.ContentAverageDissimilarity;
import util.ContentUtil;

import javax.annotation.Nonnull;
import javax.inject.Inject;
import java.util.*;

public class ContentItemScorer extends AbstractItemScorer {
	private final PreferenceSnapshot snapshot;
	private final PopModel popModel;
	private final boolean tfidf;

	@Inject
	public ContentItemScorer(@Transient @Nonnull PreferenceSnapshot snapshot, PopModel popModel, @TFIDF boolean tfidf) {
		this.snapshot = snapshot;
		this.popModel = popModel;
		this.tfidf = tfidf;
	}

	private SparseVector getUserVector(long userId) {
		Collection<IndexedPreference> preferences = snapshot.getUserRatings(userId);
		ContentAverageDissimilarity dissimilarity = ContentAverageDissimilarity.getInstance();
		Map<Long, SparseVector> map = dissimilarity.getItemContentMap();
		MutableSparseVector vector = dissimilarity.getEmptyVector();
		for (IndexedPreference pref : preferences) {
			SparseVector itemVector = map.get(pref.getItemId());
			if (tfidf) {
				itemVector = dissimilarity.toTFIDF(itemVector);
			}
			vector.add(itemVector);
		}
		return vector;
	}

	private List<Long> getRecommendations(SparseVector userVector) {
		ContentAverageDissimilarity dissimilarity = ContentAverageDissimilarity.getInstance();
		Map<Long, SparseVector> map = dissimilarity.getItemContentMap();
		LongCollection itemIds = snapshot.getItemIds();
		List<ContentContainer> list = new ArrayList<ContentContainer>();
		for (long id : itemIds) {
			SparseVector itemVector = map.get(id);
			double score;
			if (tfidf) {
				itemVector = dissimilarity.toTFIDF(itemVector);
				score = ContentUtil.getCos(userVector, itemVector);
			} else {
				score = ContentUtil.getJaccard(userVector, itemVector);
			}
			double pop = popModel.getPop(id);
			list.add(new ContentContainer(id, score, pop));
		}
		Collections.sort(list);
		//Collections.reverse(list);
		List<Long> items = new ArrayList<Long>();
		for (ContentContainer contentContainer : list) {
			items.add(contentContainer.id);
		}
		return items;
	}

	@Override
	public void score(long user, @Nonnull MutableSparseVector scores) {
		SparseVector userVector = getUserVector(user);
		List<Long> recommendations = getRecommendations(userVector);
		for (VectorEntry e : scores.view(VectorEntry.State.EITHER)) {
			scores.set(e, recommendations.indexOf(e.getKey()));
		}
	}

	private class ContentContainer implements Comparable<ContentContainer> {
		private long id;
		private Double score;
		private Double popularity;

		private ContentContainer(long id, double score, double popularity) {
			this.id = id;
			this.score = score;
			this.popularity = popularity;
		}

		@Override
		public int compareTo(ContentContainer o) {
			int res = score.compareTo(o.score);
			if (res == 0) {
				res = popularity.compareTo(o.popularity);
			}
			return res;
		}
	}
}
