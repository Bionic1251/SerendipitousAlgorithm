package lc.advanced;

import annotation.DissimilarityWeight;
import annotation.RatingPredictor;
import annotation.RelevanceWeight;
import annotation.UnpopWeight;
import lc.AlgPopModel;
import org.grouplens.lenskit.ItemScorer;
import org.grouplens.lenskit.basic.AbstractItemScorer;
import org.grouplens.lenskit.core.Transient;
import org.grouplens.lenskit.data.dao.UserEventDAO;
import org.grouplens.lenskit.data.pref.IndexedPreference;
import org.grouplens.lenskit.data.pref.PreferenceDomain;
import org.grouplens.lenskit.data.snapshot.PreferenceSnapshot;
import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.grouplens.lenskit.vectors.SparseVector;
import org.grouplens.lenskit.vectors.VectorEntry;
import pop.PopModel;
import util.AlgorithmUtil;
import util.ContentUtil;

import javax.annotation.Nonnull;
import javax.inject.Inject;
import java.util.Collection;

public class LCItemScorer extends AbstractItemScorer {
	private PopModel model;
	private final ItemScorer itemScorer;
	private final PreferenceSnapshot snapshot;
	private final double dissimilarityWeight;
	private final double unpopWeight;
	private final double relevanceWeight;
	private final PreferenceDomain domain;

	@Inject
	public LCItemScorer(PopModel model, @RatingPredictor ItemScorer itemScorer,
						@Transient @Nonnull PreferenceSnapshot snapshot, @DissimilarityWeight double dissimilarity,
						@UnpopWeight double unpop, @RelevanceWeight double relevance, PreferenceDomain domain) {
		System.out.println("LC R " + relevance + "; D " + dissimilarity + "; U " + unpop);
		this.model = model;
		this.itemScorer = itemScorer;
		this.snapshot = snapshot;
		dissimilarityWeight = dissimilarity;
		unpopWeight = unpop;
		relevanceWeight = relevance;
		this.domain = domain;
	}

	@Override
	public void score(long user, @Nonnull MutableSparseVector scores) {
		for (VectorEntry e : scores.view(VectorEntry.State.EITHER)) {
			double dissim = 0;
			if (dissimilarityWeight != 0.0) {
				dissim = getDissim(user, e.getKey());
			}
			double unpop = 1.0 - (double) model.getPop(e.getKey()) / model.getMax();
			double rating = itemScorer.score(user, e.getKey());
			double total = unpopWeight * unpop + relevanceWeight * rating / domain.getMaximum() + dissimilarityWeight * dissim;
			scores.set(e, total);
		}
	}

	private double getDissim(long userId, long itemId) {
		Collection<IndexedPreference> prefs = snapshot.getUserRatings(userId);
		SparseVector itemVec = AlgorithmUtil.itemContentMap.get(itemId);
		double dissim = 0.0;
		for (IndexedPreference p : prefs) {
			SparseVector ratedItemVec = AlgorithmUtil.itemContentMap.get(p.getItemId());
			dissim += 1.0 - ContentUtil.getCosine(ratedItemVec, itemVec);
		}
		return dissim / prefs.size();
	}
}
