package lc;

import org.grouplens.grapht.annotation.DefaultProvider;
import org.grouplens.lenskit.core.Shareable;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

@DefaultProvider(AlgPopModelBuilder.class)
@Shareable
public class AlgPopModel implements Serializable {
	private Map<Long, AlgPopModelBuilder.Container> itemMap;

	public AlgPopModel(Map<Long, AlgPopModelBuilder.Container> itemMap) {
		this.itemMap = itemMap;
	}

	public Double getPop(Long itemId) {
		if (!itemMap.containsKey(itemId)) {
			return 0.0;
		}
		return itemMap.get(itemId).getRatingNumber();
	}

	public Map<Long, AlgPopModelBuilder.Container> getItemMap() {
		return itemMap;
	}

	public List<Long> getItemList() {
		List<AlgPopModelBuilder.Container> containerList = new ArrayList<AlgPopModelBuilder.Container>(itemMap.values());
		Collections.sort(containerList);
		Collections.reverse(containerList);

		List<Long> list = new ArrayList<Long>();
		for (int i = 0; i < containerList.size(); i++) {
			list.add(containerList.get(i).getId());
		}
		return list;
	}
}
