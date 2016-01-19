package MF.lu;

import it.unimi.dsi.fastutil.longs.LongSet;
import it.unimi.dsi.fastutil.longs.LongSortedSet;
import mikera.vectorz.AVector;
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

public class LuSVDItemScorer extends AbstractItemScorer {
	private static final Logger logger = LoggerFactory.getLogger(LuSVDItemScorer.class);
	protected final LuSVDModel model;
	protected final BiasedMFKernel kernel;
	private UserEventDAO dao;
	private final ItemScorer baselineScorer;
	private final int featureCount;


	@Inject
	public LuSVDItemScorer(UserEventDAO dao, LuSVDModel model,
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

		featureCount = model.getFeatureCount();
	}

	/**
	 * Predict for a user using their preference array and history vector.
	 *
	 * @param user   The user's ID
	 * @param uprefs The user's preference vector.
	 * @param output The output vector, whose key domain is the items to predict for. It must
	 *               be initialized to the user's baseline predictions.
	 */
	private void computeScores(long user, AVector uprefs, MutableSparseVector output) {
		for (VectorEntry e : output) {
			final long item = e.getKey();
			AVector ivec = model.getItemVector(item);
			if (ivec == null) {
				// no item-vector, cannot make an informed prediction.
				// unset the baseline to note that we are not predicting for this item.
				output.unset(e);
			} else {
				double score = kernel.apply(0, uprefs, ivec);
				output.set(e, score);
			}
		}
		if(output.isEmpty()){
			System.out.println("sdsda");
		}
	}

	/**
	 * Get estimates for all a user's ratings and the target items.
	 *
	 * @param user    The user ID.
	 * @param ratings The user's ratings.
	 * @param items   The target items.
	 * @return Baseline predictions for all items either in the target set or the set of
	 *         rated items.
	 */
	private MutableSparseVector initialEstimates(long user, SparseVector ratings, LongSortedSet items) {
		LongSet allItems = LongUtils.setUnion(items, ratings.keySet());
		MutableSparseVector estimates = MutableSparseVector.create(allItems);
		baselineScorer.score(user, estimates);
		return estimates;
	}

	@Override
	public void score(long user, @Nonnull MutableSparseVector scores) {
		logger.debug("scoring {} items for user {}", scores.keyDomain().size(), user);

		UserHistory<Rating> history = dao.getEventsForUser(user, Rating.class);
		if (history == null) {
			logger.debug("found no rating history for user {}", user);
			history = History.forUser(user);
		}
		SparseVector ratings = Ratings.userRatingVector(history);

		AVector uprefs = model.getUserVector(user);
		if (uprefs == null) {
			logger.debug("no feature vector for user {}", user);
			uprefs = model.getAverageUserVector();
		}

		MutableSparseVector estimates = initialEstimates(user, ratings, scores.keyDomain());
		// propagate estimates to the output scores
		scores.set(estimates);

		// scores are the estimates, uprefs are trained up.
		computeScores(user, uprefs, scores);
	}
}
