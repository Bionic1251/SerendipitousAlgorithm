package zheng;
/*
 * LensKit, an open source recommender systems toolkit.
 * Copyright 2010-2014 LensKit Contributors.  See CONTRIBUTORS.md.
 * Work on LensKit has been funded by the National Science Foundation under
 * grants IIS 05-34939, 08-08692, 08-12148, and 10-17697.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2.1 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program; if not, write to the Free Software Foundation, Inc., 51
 * Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */


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
import org.grouplens.lenskit.knn.item.model.ItemItemModel;
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

/**
 * Do recommendations and predictions based on SVD matrix factorization.
 *
 * Recommendation is done based on folding-in.  The strategy is do a fold-in
 * operation as described in
 * <a href="http://www.grouplens.org/node/212">Sarwar et al., 2002</a> with the
 * user's ratings.
 *
 * @author <a href="http://www.grouplens.org">GroupLens Research</a>
 */
public class ZhengSVDItemScorer extends AbstractItemScorer {
	private static final Logger logger = LoggerFactory.getLogger(ZhengSVDItemScorer.class);
	protected final ZhengSVDModel model;
	protected final BiasedMFKernel kernel;
	private UserEventDAO dao;
	private final ItemScorer baselineScorer;
	private final int featureCount;

	@Inject
	public ZhengSVDItemScorer(UserEventDAO dao, ZhengSVDModel model,
							  @BaselineScorer ItemScorer baseline,
							  @Nullable PreferenceDomain dom) {
		// FIXME Unify requirement on update rule and DAO
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
