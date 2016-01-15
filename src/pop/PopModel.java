package pop;

import org.grouplens.grapht.annotation.DefaultProvider;
import org.grouplens.lenskit.core.Shareable;


import java.io.Serializable;
import java.util.List;

@DefaultProvider(PopModelBuilder.class)
@Shareable
public class PopModel implements Serializable {
	private List<Long> itemList;

	public PopModel(List<Long> itemList) {
		this.itemList = itemList;
	}

	public List<Long> getItemList() {
		return itemList;
	}
}
