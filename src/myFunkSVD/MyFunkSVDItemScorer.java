package myFunkSVD;/*
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
import mikera.vectorz.Vector;
import org.grouplens.lenskit.ItemScorer;
import org.grouplens.lenskit.baseline.BaselineScorer;
import org.grouplens.lenskit.basic.AbstractItemScorer;
import org.grouplens.lenskit.collections.LongUtils;
import org.grouplens.lenskit.data.dao.UserEventDAO;
import org.grouplens.lenskit.data.event.Rating;
import org.grouplens.lenskit.data.event.Ratings;
import org.grouplens.lenskit.data.history.History;
import org.grouplens.lenskit.data.history.RatingVectorUserHistorySummarizer;
import org.grouplens.lenskit.data.history.UserHistory;
import org.grouplens.lenskit.data.pref.PreferenceDomain;
import org.grouplens.lenskit.iterative.TrainingLoopController;
import org.grouplens.lenskit.knn.item.model.ItemItemBuildContext;
import org.grouplens.lenskit.knn.item.model.ItemItemBuildContextProvider;
import org.grouplens.lenskit.mf.funksvd.FunkSVDModel;
import org.grouplens.lenskit.mf.funksvd.RuntimeUpdate;
import org.grouplens.lenskit.mf.svd.BiasedMFKernel;
import org.grouplens.lenskit.mf.svd.DomainClampingKernel;
import org.grouplens.lenskit.mf.svd.DotProductKernel;
import org.grouplens.lenskit.transform.normalize.DefaultUserVectorNormalizer;
import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.grouplens.lenskit.vectors.SparseVector;
import org.grouplens.lenskit.vectors.VectorEntry;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import javax.inject.Inject;
import java.util.*;

/**
 * Do recommendations and predictions based on SVD matrix factorization.
 * <p/>
 * Recommendation is done based on folding-in.  The strategy is do a fold-in
 * operation as described in
 * <a href="http://www.grouplens.org/node/212">Sarwar et al., 2002</a> with the
 * user's ratings.
 *
 * @author <a href="http://www.grouplens.org">GroupLens Research</a>
 */
public class MyFunkSVDItemScorer extends AbstractItemScorer {
	private static final Logger logger = LoggerFactory.getLogger(MyFunkSVDItemScorer.class);
	protected final FunkSVDModel model;
	protected final BiasedMFKernel kernel;
	private UserEventDAO dao;
	private final ItemScorer baselineScorer;
	private final int featureCount;

	@Nullable
	private final MyFunkSVDUpdateRule rule;

	/**
	 * Construct the item scorer.
	 *
	 * @param dao      The DAO.
	 * @param model    The model.
	 * @param baseline The baseline scorer.  Be very careful when configuring a different baseline
	 *                 at runtime than at model-build time; such a configuration is unlikely to
	 *                 perform well.
	 * @param rule     The update rule, or {@code null} (the default) to only use the user features
	 *                 from the model. If provided, this update rule is used to update a user's
	 *                 feature values based on their profile when scores are requested.
	 */
	@Inject
	public MyFunkSVDItemScorer(UserEventDAO dao, FunkSVDModel model,
							   @BaselineScorer ItemScorer baseline,
							   @Nullable PreferenceDomain dom,
							   @Nullable @RuntimeUpdate MyFunkSVDUpdateRule rule) {
		// FIXME Unify requirement on update rule and DAO
		this.dao = dao;
		this.model = model;
		baselineScorer = baseline;
		this.rule = rule;

		if (dom == null) {
			kernel = new DotProductKernel();
		} else {
			kernel = new DomainClampingKernel(dom);
		}

		featureCount = model.getFeatureCount();
		updateExpectedItems();
	}

	private void updateExpectedItems() {
		ItemItemBuildContextProvider provider = new ItemItemBuildContextProvider(dao, new DefaultUserVectorNormalizer(), new RatingVectorUserHistorySummarizer());
		updateExpectedMap(provider.get());
	}

	int NUM = 100;
	private Map<Long, Map<Long, Double>> popularityMap;

	private void updateExpectedMap(ItemItemBuildContext dataContext) {
		popularityMap = new HashMap<Long, Map<Long, Double>>();
		Set<Long> userSet = new HashSet<Long>();
		List<Container> expectedItemContaners = new ArrayList<Container>();
		LongSortedSet itemSet = dataContext.getItems();
		SparseVector itemVector;
		for (Long itemId : itemSet) {
			itemVector = dataContext.itemVector(itemId);
			expectedItemContaners.add(new Container(itemId, (double) itemVector.values().size()));
			userSet.addAll(itemVector.keySet());
		}
		Collections.sort(expectedItemContaners);
		Collections.reverse(expectedItemContaners);
		Container container = expectedItemContaners.get(0);
		double maxVal = container.getRatingNumber();
		int i = 0;
		for (Container element : expectedItemContaners) {
			element.setRatingNumber(element.getRatingNumber() / maxVal);
			if(i > NUM){
				break;
			}
		}
		for (Long userId : userSet) {
			Map<Long, Double> itemMap = new HashMap<Long, Double>();
			for (Container container1 : expectedItemContaners) {
				itemMap.put(container1.getId(), container1.getRatingNumber());
			}
			popularityMap.put(userId, itemMap);
		}
	}

	@Nullable
	public MyFunkSVDUpdateRule getUpdateRule() {
		return rule;
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
				double score = kernel.apply(e.getValue(), uprefs, ivec);
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
	 * rated items.
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
			if (ratings.isEmpty() || rule == null) {
				logger.debug("no ratings or rule, bailing out on user {}", user);
				// no real work to do.
				return;
			}
			uprefs = model.getAverageUserVector();
		}

		MutableSparseVector estimates = initialEstimates(user, ratings, scores.keyDomain());
		// propagate estimates to the output scores
		scores.set(estimates);

		if (!ratings.isEmpty() && rule != null) {
			logger.debug("refreshing feature values for user {}", user);
			AVector updated = Vector.create(uprefs);
			for (int f = 0; f < featureCount; f++) {
				trainUserFeature(user, updated, ratings, estimates, f);
			}
			uprefs = updated;
		}

		// scores are the estimates, uprefs are trained up.
		computeScores(user, uprefs, scores);
	}

	private void trainUserFeature(long user, AVector uprefs, SparseVector ratings,
								  MutableSparseVector estimates, int feature) {
		assert rule != null;
		assert uprefs.length() == featureCount;
		assert feature >= 0 && feature < featureCount;

		int tailStart = feature + 1;
		int tailSize = featureCount - feature - 1;
		AVector utail = uprefs.subVector(tailStart, tailSize);
		MutableSparseVector tails = MutableSparseVector.create(ratings.keySet());
		for (VectorEntry e : tails.view(VectorEntry.State.EITHER)) {
			AVector ivec = model.getItemVector(e.getKey());
			if (ivec == null) {
				// FIXME Do this properly
				tails.set(e, 0);
			} else {
				ivec = ivec.subVector(tailStart, tailSize);
				tails.set(e, utail.dotProduct(ivec));
			}
		}

		double rmse = Double.MAX_VALUE;
		TrainingLoopController controller = rule.getTrainingLoopController();
		while (controller.keepTraining(rmse)) {
			rmse = doFeatureIteration(user, uprefs, ratings, estimates, feature, tails);
		}
	}

	private double doFeatureIteration(long user, AVector uprefs,
									  SparseVector ratings, MutableSparseVector estimates,
									  int feature, SparseVector itemTails) {
		assert rule != null;

		MyFunkSVDUpdater updater = rule.createUpdater();
		for (VectorEntry e : ratings) {
			final long iid = e.getKey();
			final AVector ivec = model.getItemVector(iid);
			if (ivec == null) {
				continue;
			}

			Map<Long, Double> itemMap = popularityMap.get(user);
			double val = itemMap.get(iid);

			updater.prepare(feature, e.getValue(), estimates.get(iid),
					uprefs.get(feature), ivec.get(feature), itemTails.get(iid), val + 1);
			// Step 4: update user preferences
			uprefs.addAt(feature, updater.getUserFeatureUpdate());
		}
		return updater.getRMSE();
	}

	private class Container implements Comparable<Container> {
		private Long id;
		private Double ratingNumber;

		private Container(Long id, Double ratingNumber) {
			this.id = id;
			this.ratingNumber = ratingNumber;
		}

		@Override
		public boolean equals(Object obj) {
			Container container = (Container) obj;
			return container.id.equals(id);
		}

		@Override
		public int compareTo(Container o) {
			return ratingNumber.compareTo(o.ratingNumber);
		}

		public Long getId() {
			return id;
		}

		public void setRatingNumber(Double ratingNumber) {
			this.ratingNumber = ratingNumber;
		}

		public Double getRatingNumber() {
			return ratingNumber;
		}
	}
}
