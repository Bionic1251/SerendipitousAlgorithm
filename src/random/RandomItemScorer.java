package random;

import it.unimi.dsi.fastutil.longs.LongCollection;
import it.unimi.dsi.fastutil.longs.LongSortedSet;
import org.grouplens.lenskit.basic.AbstractItemScorer;
import org.grouplens.lenskit.core.Transient;
import org.grouplens.lenskit.data.snapshot.PreferenceSnapshot;
import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.grouplens.lenskit.vectors.VectorEntry;

import javax.annotation.Nonnull;
import javax.inject.Inject;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

public class RandomItemScorer extends AbstractItemScorer {
	private List<Long> itemList;

	@Inject
	public RandomItemScorer(@Transient @Nonnull PreferenceSnapshot snapshot) {
		LongCollection itemIds = snapshot.getItemIds();
		itemList = new ArrayList<Long>();
		for(long id : itemIds){
			itemList.add(id);
		}
		Collections.shuffle(itemList);
	}

	@Override
	public void score(long user, @Nonnull MutableSparseVector scores) {
		for (VectorEntry e : scores.view(VectorEntry.State.EITHER)) {
			scores.set(e.getKey(), itemList.indexOf(e.getKey()));
		}
	}
}
