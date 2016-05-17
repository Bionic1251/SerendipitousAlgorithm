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
package evaluationMetric;

import it.unimi.dsi.fastutil.longs.LongArrayList;
import it.unimi.dsi.fastutil.longs.LongIterator;
import it.unimi.dsi.fastutil.longs.LongList;
import org.grouplens.lenskit.Recommender;
import org.grouplens.lenskit.eval.Attributed;
import org.grouplens.lenskit.eval.data.traintest.TTDataSet;
import org.grouplens.lenskit.eval.metrics.AbstractMetric;
import org.grouplens.lenskit.eval.metrics.ResultColumn;
import org.grouplens.lenskit.eval.metrics.topn.ItemSelector;
import org.grouplens.lenskit.eval.metrics.topn.TopNMetricBuilder;
import org.grouplens.lenskit.eval.traintest.TestUser;
import org.grouplens.lenskit.scored.ScoredId;
import org.grouplens.lenskit.util.statistics.MeanAccumulator;
import org.grouplens.lenskit.vectors.SparseVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

import static java.lang.Math.log;

/**
 * @author <a href="http://www.grouplens.org">GroupLens Research</a>
 */
public class AggregateNDCGTopNMetric extends AbstractMetric<MeanAccumulator, AggregateNDCGTopNMetric.AggregateResult, AggregateNDCGTopNMetric.AggregateResult> {
	private static final Logger logger = LoggerFactory.getLogger(AggregateNDCGTopNMetric.class);

	private final int LIST_SIZE = 30;
	private final ItemSelector candidates;
	private final ItemSelector exclude;
	private final String prefix;
	private final String suffix;

	private MeanAccumulator context1;
	private MeanAccumulator context5;
	private MeanAccumulator context10;
	private MeanAccumulator context15;
	private MeanAccumulator context20;
	private MeanAccumulator context25;
	private MeanAccumulator context30;

	/**
	 * Construct a new nDCG Top-N metric.
	 *
	 * @param pre        the prefix label for this evaluation, or {@code null} for no prefix.
	 * @param sfx        the suffix label for this evaluation, or {@code null} for no suffix.
	 * @param candidates The candidate selector.
	 * @param exclude    The exclude selector.
	 */
	public AggregateNDCGTopNMetric(String pre, String sfx, ItemSelector candidates, ItemSelector exclude) {
		super(AggregateResult.class, AggregateResult.class);
		suffix = sfx;
		prefix = pre;
		this.candidates = candidates;
		this.exclude = exclude;
	}

	@Override
	public MeanAccumulator createContext(Attributed algo, TTDataSet ds, Recommender rec) {
		context1 = new MeanAccumulator();
		context5 = new MeanAccumulator();
		context10 = new MeanAccumulator();
		context15 = new MeanAccumulator();
		context20 = new MeanAccumulator();
		context25 = new MeanAccumulator();
		context30 = new MeanAccumulator();
		return new MeanAccumulator();
	}

	@Override
	protected String getPrefix() {
		return prefix;
	}

	@Override
	protected String getSuffix() {
		return suffix;
	}

	/**
	 * Compute the DCG of a list of items with respect to a value vector.
	 */
	static double computeDCG(LongList items, SparseVector values) {
		final double lg2 = log(2);

		double gain = 0;
		int rank = 0;

		LongIterator iit = items.iterator();
		while (iit.hasNext()) {
			final long item = iit.nextLong();
			final double v = values.get(item, 0);
			rank++;
			if (rank < 2) {
				gain += v;
			} else {
				gain += v * lg2 / log(rank);
			}
		}

		return gain;
	}

	@Override
	public AggregateResult doMeasureUser(TestUser user, MeanAccumulator context) {
		List<ScoredId> recommendations = user.getRecommendations(30, candidates, exclude);
		if (recommendations == null || recommendations.isEmpty()) {
			context1.add(0.0);
			context5.add(0.0);
			context10.add(0.0);
			context15.add(0.0);
			context20.add(0.0);
			context25.add(0.0);
			context30.add(0.0);
			return null;
		}
		double ndcg1 = measureUser(user, context1, recommendations, 1);
		double ndcg5 = measureUser(user, context5, recommendations, 5);
		double ndcg10 = measureUser(user, context10, recommendations, 10);
		double ndcg15 = measureUser(user, context15, recommendations, 15);
		double ndcg20 = measureUser(user, context20, recommendations, 20);
		double ndcg25 = measureUser(user, context25, recommendations, 25);
		double ndcg30 = measureUser(user, context30, recommendations, 30);
		return new AggregateResult(ndcg1, ndcg5, ndcg10, ndcg15, ndcg20, ndcg25, ndcg30);
	}

	public double measureUser(TestUser user, MeanAccumulator context, List<ScoredId> recommendations, int listSize) {
		if(recommendations.size() > listSize){
			recommendations = new ArrayList<ScoredId>(recommendations.subList(0, listSize));
		}
		SparseVector ratings = user.getTestRatings();
		LongList ideal = ratings.keysByValue(true);
		if (ideal.size() > listSize) {
			ideal = ideal.subList(0, listSize);
		}
		double idealGain = computeDCG(ideal, ratings);

		LongList actual = new LongArrayList(recommendations.size());
		for (ScoredId id : recommendations) {
			actual.add(id.getId());
		}
		double gain = computeDCG(actual, ratings);

		double score = gain / idealGain;

		context.add(score);
		return score;
	}

	@Override
	protected AggregateResult getTypedResults(MeanAccumulator context) {
		return new AggregateResult(context1.getMean(), context5.getMean(), context10.getMean(), context15.getMean(), context20.getMean(), context25.getMean(), context30.getMean());
	}

	public static class AggregateResult {
		@ResultColumn("nDCG1")
		public final double nDCG1;

		@ResultColumn("nDCG5")
		public final double nDCG5;

		@ResultColumn("nDCG10")
		public final double nDCG10;

		@ResultColumn("nDCG15")
		public final double nDCG15;

		@ResultColumn("nDCG20")
		public final double nDCG20;

		@ResultColumn("nDCG25")
		public final double nDCG25;

		@ResultColumn("nDCG30")
		public final double nDCG30;

		public AggregateResult(double nDCG1, double nDCG5, double nDCG10, double nDCG15, double nDCG20, double nDCG25, double nDCG30) {
			this.nDCG1 = nDCG1;
			this.nDCG5 = nDCG5;
			this.nDCG10 = nDCG10;
			this.nDCG15 = nDCG15;
			this.nDCG20 = nDCG20;
			this.nDCG25 = nDCG25;
			this.nDCG30 = nDCG30;
		}
	}

	/**
	 * @author <a href="http://www.grouplens.org">GroupLens Research</a>
	 */
	public static class Builder extends TopNMetricBuilder<AggregateNDCGTopNMetric> {
		@Override
		public AggregateNDCGTopNMetric build() {
			return new AggregateNDCGTopNMetric(prefix, suffix, candidates, exclude);
		}
	}

}
