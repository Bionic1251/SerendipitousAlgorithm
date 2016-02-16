package alg.pop;

import annotation.DissimilarityWeight;
import annotation.UnpopWeight;
import annotation.RatingPredictor;
import annotation.RelevanceWeight;
import org.grouplens.lenskit.ItemScorer;
import org.grouplens.lenskit.basic.AbstractItemScorer;
import org.grouplens.lenskit.core.Transient;
import org.grouplens.lenskit.data.dao.UserEventDAO;
import org.grouplens.lenskit.data.pref.IndexedPreference;
import org.grouplens.lenskit.data.snapshot.PreferenceSnapshot;
import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.grouplens.lenskit.vectors.SparseVector;
import org.grouplens.lenskit.vectors.VectorEntry;
import util.AlgorithmUtil;
import util.ContentUtil;

import javax.annotation.Nonnull;
import javax.inject.Inject;
import java.util.Collection;
import java.util.HashSet;
import java.util.Set;

public class AlgPopItemScorer extends AbstractItemScorer {
	private AlgPopModel model;
	private UserEventDAO dao;
	private final ItemScorer itemScorer;
	private final PreferenceSnapshot snapshot;
	private final double dissimilarityWeight;
	private final double unpopWeight;
	private final double relevanceWeight;

	@Inject
	public AlgPopItemScorer(AlgPopModel model, UserEventDAO dao, @RatingPredictor ItemScorer itemScorer,
							@Transient @Nonnull PreferenceSnapshot snapshot, @DissimilarityWeight double dissimilarity,
							@UnpopWeight double unpop, @RelevanceWeight double relevance) {
		System.out.println("Alg unpop " + unpop + " dissim " + dissimilarity + " rel " + relevance);
		this.dao = dao;
		this.model = model;
		this.itemScorer = itemScorer;
		this.snapshot = snapshot;
		dissimilarityWeight = dissimilarity;
		unpopWeight = unpop;
		relevanceWeight = relevance;
	}

	@Override
	public void score(long user, @Nonnull MutableSparseVector scores) {
		for (VectorEntry e : scores.view(VectorEntry.State.EITHER)) {
			double dissim = 0;
			if (dissimilarityWeight != 0.0) {
				dissim = getDissim(user, e.getKey());
			}
			double unpop = 1 - model.getPop(e.getKey());
			double rating = itemScorer.score(user, e.getKey());
			double total = unpopWeight * unpop + relevanceWeight * rating / 5 + dissimilarityWeight * dissim;
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
