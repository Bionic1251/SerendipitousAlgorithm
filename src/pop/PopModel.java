package pop;

import org.grouplens.grapht.annotation.DefaultProvider;
import org.grouplens.lenskit.core.Shareable;


import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

@DefaultProvider(PopModelBuilder.class)
@Shareable
public class PopModel implements Serializable {
	private Map<Long, PopModelBuilder.Container> itemMap;

	public PopModel(Map<Long, PopModelBuilder.Container> itemMap) {
		this.itemMap = itemMap;
	}

	public Integer getPop(Long itemId){
		if(!itemMap.containsKey(itemId)){
			return 0;
		}
		return itemMap.get(itemId).getRatingNumber();
	}

	public List<Long> getItemList() {
		List<PopModelBuilder.Container> containerList = new ArrayList<PopModelBuilder.Container>(itemMap.values());
		Collections.sort(containerList);
		//Collections.reverse(containerList);

		List<Long> list = new ArrayList<Long>();
		for(int i = 0; i < containerList.size(); i++){
			list.add(containerList.get(i).getId());
		}
		return list;
	}
}
