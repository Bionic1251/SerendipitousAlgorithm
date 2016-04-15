package lc.investigation;


import org.grouplens.lenskit.core.Transient;
import org.grouplens.lenskit.data.snapshot.PreferenceSnapshot;

import pop.PopModel;


import javax.annotation.Nonnull;
import javax.inject.Inject;


public class NonPersInvestigationItemScorer extends InvestigationItemScorer {
	private WeightTriple weights;

	@Inject
	public NonPersInvestigationItemScorer(PopModel popModel, @Transient @Nonnull PreferenceSnapshot snapshot) {
		super(popModel, snapshot, "non-pers.tsv");
	}

	@Override
	protected void train() {
		super.train();
		trainWeights();
	}

	private void trainWeights() {
		Maximizer maximizer = new Maximizer(serR, serD, serU, unserR, unserD, unserU);
		maximizer.optimize();
		weights = maximizer.getWeights();
		print(0l, 0, weights, serR, serD, serU, unserR, unserD, unserU);
		close();
	}
}
