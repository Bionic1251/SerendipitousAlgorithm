package evaluationMetric;
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
import org.grouplens.lenskit.Recommender;
import org.grouplens.lenskit.eval.Attributed;
import org.grouplens.lenskit.eval.data.traintest.TTDataSet;
import org.grouplens.lenskit.eval.metrics.AbstractMetric;
import org.grouplens.lenskit.eval.metrics.ResultColumn;
import org.grouplens.lenskit.eval.metrics.topn.ItemSelector;
import org.grouplens.lenskit.eval.metrics.topn.ItemSelectors;
import org.grouplens.lenskit.eval.metrics.topn.PrecisionRecallTopNMetric;
import org.grouplens.lenskit.eval.metrics.topn.TopNMetricBuilder;
import org.grouplens.lenskit.eval.traintest.TestUser;
import org.grouplens.lenskit.scored.ScoredId;
import org.hamcrest.Matchers;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * A metric to compute the precision and recall of a recommender given a
 * set of candidate items to recommend from and a set of desired items.  The aggregate results are
 * means of the user results.
 * <p/>
 * This can be used to compute metrics like fallout (probability that a
 * recommendation is bad) by configuring bad items as the test item set.
 *
 * @author <a href="http://www.grouplens.org">GroupLens Research</a>
 */
public class AggregatePrecisionRecallTopNMetric extends AbstractMetric<AggregatePrecisionRecallTopNMetric.Context, AggregatePrecisionRecallTopNMetric.AggregateResult, AggregatePrecisionRecallTopNMetric.AggregateResult> {
	private static final Logger logger = LoggerFactory.getLogger(AggregatePrecisionRecallTopNMetric.class);
	private final String prefix;
	private final String suffix;
	private final ItemSelector candidates;
	private final ItemSelector exclude;
	private final ItemSelector goodItems;

	private Context context1;
	private Context context5;
	private Context context10;
	private Context context15;
	private Context context20;
	private Context context25;
	private Context context30;

	/**
	 * Construct a new recall and precision top n metric
	 *
	 * @param pre        the prefix label for this evaluation, or {@code null} for no prefix.
	 * @param sfx        the suffix label for this evaluation, or {@code null} for no suffix.
	 * @param candidates The candidate selector, provides a list of items which can be recommended
	 * @param exclude    The exclude selector, provides a list of items which must not be recommended
	 *                   (These items are removed from the candidate items to form the final candidate set)
	 * @param goodItems  The list of items to consider "true positives", all other items will be treated
	 *                   as "false positives".
	 */
	public AggregatePrecisionRecallTopNMetric(String pre, String sfx, ItemSelector candidates, ItemSelector exclude, ItemSelector goodItems) {
		super(AggregateResult.class, AggregateResult.class);
		prefix = pre;
		suffix = sfx;
		this.candidates = candidates;
		this.exclude = exclude;
		this.goodItems = goodItems;
	}

	@Override
	protected String getPrefix() {
		return prefix;
	}

	@Override
	protected String getSuffix() {
		return suffix;
	}

	@Override
	public Context createContext(Attributed algo, TTDataSet ds, Recommender rec) {
		context1 = new Context();
		context5 = new Context();
		context10 = new Context();
		context15 = new Context();
		context20 = new Context();
		context25 = new Context();
		context30 = new Context();
		return new Context();
	}

	@Override
	public AggregateResult doMeasureUser(TestUser user, Context context) {
		List<ScoredId> recs = user.getRecommendations(30, candidates, exclude);
		if (recs == null || recs.isEmpty()) {
			context1.addUser(0,0);
			context5.addUser(0,0);
			context10.addUser(0,0);
			context15.addUser(0,0);
			context20.addUser(0,0);
			context25.addUser(0,0);
			context30.addUser(0,0);
			logger.warn("no recommendations for user {}", user.getUserId());
			return new AggregateResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
		}

		LongSet items = goodItems.select(user);
		if (items.isEmpty()) {
			logger.warn("no good items for user {}", user.getUserId());
		}
		UsualResult r1 = measureUser(user, context1, 1, recs, items);
		UsualResult r5 = measureUser(user, context5, 5, recs, items);
		UsualResult r10 = measureUser(user, context10, 10, recs, items);
		UsualResult r15 = measureUser(user, context15, 15, recs, items);
		UsualResult r20 = measureUser(user, context20, 20, recs, items);
		UsualResult r25 = measureUser(user, context25, 25, recs, items);
		UsualResult r30 = measureUser(user, context30, 30, recs, items);
		return new AggregateResult(r1.precision, r1.recall, r5.precision, r5.recall, r10.precision, r10.recall, r15.precision, r15.recall, r20.precision, r20.recall, r25.precision, r25.recall, r30.precision, r30.recall);
	}

	public UsualResult measureUser(TestUser user, Context context, int listSize, List<ScoredId> recs, LongSet items) {
		if (recs.size() > listSize) {
			recs = new ArrayList<ScoredId>(recs.subList(0, listSize));
		}
		int tp = 0;

		logger.debug("searching for {} good items among {} recommendations for {}",
				items.size(), recs.size(), user.getUserId());
		for (ScoredId s : recs) {
			if (items.contains(s.getId())) {
				tp += 1;
			}
		}

		if (items.size() > 0 && recs.size() > 0) {
			// if both the items set and recommendations are non-empty (no division by 0).
			double precision = (double) tp / recs.size();
			double recall = (double) tp / items.size();
			context.addUser(precision, recall);
			return new UsualResult(precision, recall);
		} else {
			return new UsualResult(0.0, 0.0);
		}
	}

	@Override
	protected AggregateResult getTypedResults(Context context) {
		UsualResult r1 = context1.finish();
		UsualResult r5 = context5.finish();
		UsualResult r10 = context10.finish();
		UsualResult r15 = context15.finish();
		UsualResult r20 = context20.finish();
		UsualResult r25 = context25.finish();
		UsualResult r30 = context30.finish();
		return new AggregateResult(r1.precision, r1.recall, r5.precision, r5.recall, r10.precision, r10.recall, r15.precision, r15.recall, r20.precision, r20.recall, r25.precision, r25.recall, r30.precision, r30.recall);
	}

	public static class AggregateResult {
		@ResultColumn("Precision1")
		public final double precision1;
		@ResultColumn("Recall1")
		public final double recall1;

		@ResultColumn("Precision5")
		public final double precision5;
		@ResultColumn("Recall5")
		public final double recall5;

		@ResultColumn("Precision10")
		public final double precision10;
		@ResultColumn("Recall10")
		public final double recall10;

		@ResultColumn("Precision15")
		public final double precision15;
		@ResultColumn("Recall15")
		public final double recall15;

		@ResultColumn("Precision20")
		public final double precision20;
		@ResultColumn("Recall20")
		public final double recall20;

		@ResultColumn("Precision25")
		public final double precision25;
		@ResultColumn("Recall25")
		public final double recall25;

		@ResultColumn("Precision30")
		public final double precision30;
		@ResultColumn("Recall30")
		public final double recall30;

		public AggregateResult(double precision1, double recall1, double precision5, double recall5, double precision10, double recall10, double precision15, double recall15, double precision20, double recall20, double precision25, double recall25, double precision30, double recall30) {
			this.precision1 = precision1;
			this.recall1 = recall1;
			this.precision5 = precision5;
			this.recall5 = recall5;
			this.precision10 = precision10;
			this.recall10 = recall10;
			this.precision15 = precision15;
			this.recall15 = recall15;
			this.precision20 = precision20;
			this.recall20 = recall20;
			this.precision25 = precision25;
			this.recall25 = recall25;
			this.precision30 = precision30;
			this.recall30 = recall30;
		}
	}

	public static class UsualResult {
		public final double precision;
		public final double recall;

		public UsualResult(double prec, double rec) {
			precision = prec;
			recall = rec;
		}

		public double getPrecision() {
			return precision;
		}

		public double getRecall() {
			return recall;
		}
	}

	public class Context {
		double totalPrecision = 0;
		double totalRecall = 0;
		int nusers = 0;

		private void addUser(double prec, double rec) {
			totalPrecision += prec;
			totalRecall += rec;
			nusers += 1;
		}

		public UsualResult finish() {
			if (nusers > 0) {
				return new UsualResult(totalPrecision / nusers, totalRecall / nusers);
			} else {
				return null;
			}
		}
	}

	/**
	 * @author <a href="http://www.grouplens.org">GroupLens Research</a>
	 */
	public static class Builder extends TopNMetricBuilder<AggregatePrecisionRecallTopNMetric> {
		private ItemSelector goodItems = ItemSelectors.testRatingMatches(Matchers.greaterThanOrEqualTo(4.0d));

		public Builder() {
			// override the default candidate items with a more reasonable set.
			setCandidates(ItemSelectors.allItems());
		}

		public ItemSelector getGoodItems() {
			return goodItems;
		}

		/**
		 * Set the set of items that will be considered &lsquo;good&rsquo; by the evaluation.
		 *
		 * @param goodItems A selector for good items.
		 */
		public void setGoodItems(ItemSelector goodItems) {
			this.goodItems = goodItems;
		}

		@Override
		public AggregatePrecisionRecallTopNMetric build() {
			return new AggregatePrecisionRecallTopNMetric(prefix, suffix, candidates, exclude, goodItems);
		}
	}

}

