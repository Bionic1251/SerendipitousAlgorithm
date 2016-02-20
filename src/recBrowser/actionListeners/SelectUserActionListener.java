package recBrowser.actionListeners;

import evaluationMetric.Container;
import org.grouplens.lenskit.vectors.SparseVector;
import org.grouplens.lenskit.vectors.VectorEntry;
import recBrowser.BrowserSettings;
import recBrowser.TestTrainReader;
import recBrowser.UserRecReader;
import util.ContentAverageDissimilarity;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.text.NumberFormat;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

public class SelectUserActionListener implements ActionListener {
	private final JPanel userPanel;
	private final JList userList;
	private final JList algList;

	public SelectUserActionListener(JList algList, JList userList, JPanel userPanel) {
		this.algList = algList;
		this.userList = userList;
		this.userPanel = userPanel;
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		ContentAverageDissimilarity dissimilarity = ContentAverageDissimilarity.getInstance();
		Map<Long, SparseVector> contentMap = dissimilarity.getItemContentMap();
		Iterator<SparseVector> iterator = contentMap.values().iterator();
		int featureNumber = iterator.next().view(VectorEntry.State.EITHER).size();
		int[] sum = new int[featureNumber];

		String userId = (String) userList.getSelectedValue();
		String algName = (String) algList.getSelectedValue();
		TestTrainReader trainReader = new TestTrainReader(userId);
		trainReader.readFile(BrowserSettings.trainFilePath);
		Map<Long, Double> trainItemMap = trainReader.getItemMap();
		TestTrainReader testReader = new TestTrainReader(userId);
		testReader.readFile(BrowserSettings.testFilePath);
		Map<Long, Double> testItemMap = testReader.getItemMap();
		int shift = 3;
		String[] columns = new String[shift + featureNumber];
		columns[0] = "Item";
		columns[1] = "Rating";
		columns[2] = "Score";
		for (int i = shift; i < featureNumber + shift; i++) {
			columns[i] = (i - shift) + "";
		}
		UserRecReader recReader = new UserRecReader(algName, userId);
		recReader.readFile(BrowserSettings.outFilePath);
		List<Container<Double>> scores = recReader.getScoreList();
		NumberFormat numberFormat = NumberFormat.getNumberInstance();
		numberFormat.setMaximumFractionDigits(3);
		String[][] data = new String[scores.size() + trainItemMap.size() + 1][shift + featureNumber];
		int i = 0;
		for (Map.Entry<Long, Double> entry : trainItemMap.entrySet()) {
			data[i][0] = entry.getKey().toString();
			data[i][1] = entry.getValue().toString();
			data[i][2] = "";
			printContentData(contentMap.get(entry.getKey()), sum, data[i], shift);
			i++;
		}
		data[i][0] = "";
		data[i][1] = "";
		data[i][2] = "";
		for (int j = shift; j < featureNumber + shift; j++) {
			data[i][j] = sum[j - shift] + "";
		}
		i++;
		for (int j = 0; j < scores.size(); j++) {
			data[i][0] = numberFormat.format(scores.get(j).getId());
			data[i][2] = numberFormat.format(scores.get(j).getValue());
			if (testItemMap.containsKey(scores.get(j).getId())) {
				data[i][1] = testItemMap.get(scores.get(j).getId()).toString();
			} else {
				data[i][1] = "";
			}
			printContentData(contentMap.get(scores.get(j).getId()), sum, data[i], shift);
			i++;
		}

		userPanel.removeAll();
		JTable jTable = new JTable(data, columns);
		JScrollPane scrollPane = new JScrollPane(jTable);
		jTable.setFillsViewportHeight(true);
		userPanel.add(scrollPane);
		userPanel.updateUI();
	}

	private void printContentData(SparseVector vector, int[] sum, String[] data, int shift) {
		for (int i = 0; i < sum.length; i++) {
			if (vector.containsKey(i)) {
				sum[i]++;
				data[i + shift] = "1";
			} else {
				data[i + shift] = "0";
			}
		}
	}
}
