package recBrowser.actionListeners;

import evaluationMetric.Container;
import recBrowser.BrowserSettings;
import recBrowser.UserRecReader;

import javax.swing.*;
import javax.swing.table.JTableHeader;
import javax.swing.table.TableColumn;
import javax.swing.table.TableColumnModel;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.text.NumberFormat;
import java.util.List;

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
		String userId = (String) userList.getSelectedValue();
		String algName = (String) algList.getSelectedValue();
		UserRecReader recReader = new UserRecReader(algName, userId);
		recReader.readFile(BrowserSettings.outFilePath);
		List<Container<Double>> scores = recReader.getScoreList();
		String[] columns = {"Item", "Score"};
		NumberFormat numberFormat = NumberFormat.getNumberInstance();
		numberFormat.setMaximumFractionDigits(3);
		String[][] data = new String[scores.size()][2];
		for (int i = 0; i < scores.size(); i++) {
			data[i][0] = numberFormat.format(scores.get(i).getId());
			data[i][1] = numberFormat.format(scores.get(i).getValue());
		}

		userPanel.removeAll();
		JTable jTable = new JTable(data, columns);
		JScrollPane scrollPane = new JScrollPane(jTable);
		jTable.setFillsViewportHeight(true);
		userPanel.add(scrollPane);
		userPanel.updateUI();
	}
}
