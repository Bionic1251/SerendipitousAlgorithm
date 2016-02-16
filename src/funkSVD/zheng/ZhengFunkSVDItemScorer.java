package funkSVD.zheng;

import it.unimi.dsi.fastutil.longs.LongSet;
import it.unimi.dsi.fastutil.longs.LongSortedSet;
import mikera.vectorz.AVector;
import mikera.vectorz.Vector;
import org.grouplens.lenskit.ItemScorer;
import org.grouplens.lenskit.baseline.BaselineScorer;
import org.grouplens.lenskit.basic.AbstractItemScorer;
import org.grouplens.lenskit.collections.LongUtils;
import org.grouplens.lenskit.data.dao.UserEventDAO;
import org.grouplens.lenskit.data.event.Rating;
import org.grouplens.lenskit.data.event.Ratings;
import org.grouplens.lenskit.data.history.History;
import org.grouplens.lenskit.data.history.UserHistory;
import org.grouplens.lenskit.data.pref.PreferenceDomain;
import org.grouplens.lenskit.iterative.TrainingLoopController;
import org.grouplens.lenskit.mf.funksvd.RuntimeUpdate;
import org.grouplens.lenskit.mf.svd.BiasedMFKernel;
import org.grouplens.lenskit.mf.svd.DomainClampingKernel;
import org.grouplens.lenskit.mf.svd.DotProductKernel;
import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.grouplens.lenskit.vectors.SparseVector;
import org.grouplens.lenskit.vectors.VectorEntry;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import javax.inject.Inject;

public class ZhengFunkSVDItemScorer extends AbstractItemScorer {
	protected final ZhengFunkSVDModel model;
	protected final BiasedMFKernel kernel;
	private UserEventDAO dao;
	private final ItemScorer baselineScorer;

	@Inject
	public ZhengFunkSVDItemScorer(UserEventDAO dao, ZhengFunkSVDModel model,
								  @BaselineScorer ItemScorer baseline,
								  @Nullable PreferenceDomain dom) {
		this.dao = dao;
		this.model = model;
		baselineScorer = baseline;

		if (dom == null) {
			kernel = new DotProductKernel();
		} else {
			kernel = new DomainClampingKernel(dom);
		}
	}

	private MutableSparseVector initialEstimates(long user, SparseVector ratings, LongSortedSet items) {
		LongSet allItems = LongUtils.setUnion(items, ratings.keySet());
		MutableSparseVector estimates = MutableSparseVector.create(allItems);
		baselineScorer.score(user, estimates);
		return estimates;
	}

	@Override
	public void score(long user, @Nonnull MutableSparseVector scores) {
		UserHistory<Rating> history = dao.getEventsForUser(user, Rating.class);
		if (history == null) {
			history = History.forUser(user);
		}
		SparseVector ratings = Ratings.userRatingVector(history);

		AVector uprefs = model.getUserVector(user);
		if (uprefs == null) {
			if (ratings.isEmpty()) {
				return;
			}
			uprefs = model.getAverageUserVector();
		}

		MutableSparseVector estimates = initialEstimates(user, ratings, scores.keyDomain());
		scores.set(estimates);

		for (VectorEntry e : scores) {
			final long item = e.getKey();
			AVector ivec = model.getItemVector(item);
			if (ivec == null) {
				scores.unset(e);
			} else {
				double score = e.getValue() + uprefs.dotProduct(ivec);
				scores.set(e, score);
			}
		}
	}
}
