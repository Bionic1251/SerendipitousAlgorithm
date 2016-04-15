package lc.investigation;

import org.grouplens.lenskit.core.Transient;
import org.grouplens.lenskit.data.snapshot.PreferenceSnapshot;
import pop.PopModel;

import javax.annotation.Nonnull;
import javax.inject.Inject;
import java.util.Collection;

public class PersInvestigationItemScorer extends InvestigationItemScorer {

	@Inject
	public PersInvestigationItemScorer(PopModel popModel, @Transient @Nonnull PreferenceSnapshot snapshot) {
		super(popModel, snapshot, "pers.tsv");
	}

	@Override
	protected void countForEachUser(Long userId) {
		serR = 0;
		serD = 0;
		serU = 0;
		unserR = 0;
		unserD = 0;
		unserU = 0;
		super.countForEachUser(userId);
		Maximizer maximizer = new Maximizer(serR, serD, serU, unserR, unserD, unserU);
		maximizer.optimize();
		WeightTriple triple = maximizer.getWeights();
		Collection profile = snapshot.getUserRatings(userId);
		print(userId, profile.size(), triple, serR, serD, serU, unserR, unserD, unserU);
	}

	@Override
	protected void train() {
		super.train();
		close();
	}
}
