package content;

import annotation.RatingPredictor;
import annotation.TFIDF;
import evaluationMetric.Container;
import org.grouplens.lenskit.ItemScorer;
import org.grouplens.lenskit.basic.AbstractItemScorer;
import org.grouplens.lenskit.core.Transient;
import org.grouplens.lenskit.data.pref.IndexedPreference;
import org.grouplens.lenskit.data.snapshot.PreferenceSnapshot;
import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.grouplens.lenskit.vectors.SparseVector;
import org.grouplens.lenskit.vectors.VectorEntry;
import pop.PopModel;
import util.ContentAverageDissimilarity;
import util.Settings;

import javax.annotation.Nonnull;
import javax.inject.Inject;
import java.util.*;

public class SerContentItemScorer extends AbstractItemScorer {
	private final PreferenceSnapshot snapshot;
	private final PopModel popModel;
	private final boolean tfidf;
	private final ItemScorer baseline;
	Map<Long, Set<Long>> popMap;

	@Inject
	public SerContentItemScorer(@Transient @Nonnull PreferenceSnapshot snapshot, PopModel popModel, @TFIDF boolean tfidf, @RatingPredictor ItemScorer baseline) {
		this.snapshot = snapshot;
		this.popModel = popModel;
		this.tfidf = tfidf;
		this.baseline = baseline;
		getDistribution();
	}

	private void getDistribution() {
		ContentAverageDissimilarity dissimilarity = ContentAverageDissimilarity.getInstance();
		Map<Long, SparseVector> map = dissimilarity.getItemContentMap();
		map = cleanMap(map);
		Collection<Long> keySet = dissimilarity.getEmptyVector().keySet();
		popMap = new HashMap<Long, Set<Long>>();
		for (long key : keySet) {
			List<Long> items = getItemsByGenre(key, map);
			List<Container<Integer>> containerList = new ArrayList<Container<Integer>>();
			for (long itemId : items) {
				containerList.add(new Container<Integer>(itemId, popModel.getPop(itemId)));
			}
			Collections.sort(containerList);
			Collections.reverse(containerList);
			int frac = (int) (Settings.FRACTION * (double) containerList.size());
			containerList = containerList.subList(0, frac);
			Set<Long> popSet = new HashSet<Long>();
			for (Container<Integer> container : containerList) {
				popSet.add(container.getId());
			}
			popMap.put(key, popSet);
		}
	}

	private SparseVector getUserVector(long userId) {
		Collection<IndexedPreference> preferences = snapshot.getUserRatings(userId);
		ContentAverageDissimilarity dissimilarity = ContentAverageDissimilarity.getInstance();
		Map<Long, SparseVector> map = dissimilarity.getItemContentMap();
		MutableSparseVector vector = dissimilarity.getEmptyVector();
		for (IndexedPreference pref : preferences) {
			/*if (pref.getValue() <= Settings.R_THRESHOLD) {
				continue;
			}*/
			SparseVector itemVector = map.get(pref.getItemId());
			if (tfidf) {
				itemVector = dissimilarity.toTFIDF(itemVector);
			}
			vector.add(itemVector);
		}
		return vector;
	}

	private List<Long> getRecommendations(long userId, SparseVector userVector) {
		ContentAverageDissimilarity dissimilarity = ContentAverageDissimilarity.getInstance();
		Map<Long, SparseVector> map = dissimilarity.getItemContentMap();
		map = cleanMap(map);
		Set<Long> newFeatures = getGenreSet(userVector);
		SparseVector prediction = baseline.score(userId, snapshot.getItemIds());
		Map<Long, LinkedList<Container<Double>>> genresMap = new HashMap<Long, LinkedList<Container<Double>>>();
		for (long feature : newFeatures) {
			LinkedList<Long> list = getItemsByGenre(feature, map);
			cleanList(list, feature);
			LinkedList<Container<Double>> containerList = toContainerList(list, prediction);
			genresMap.put(feature, containerList);
		}
		List<Long> list = getMixedList(genresMap);
		return list;
	}

	private boolean isPop(long itemId) {
		for (long key : popMap.keySet()) {
			if (popMap.get(key).contains(itemId)) {
				return true;
			}
		}
		return false;
	}

	private void cleanList(LinkedList<Long> list, long genre) {
		Iterator<Long> iterator = list.iterator();
		while (iterator.hasNext()) {
			Long itemId = iterator.next();
			if (isPop(itemId)) {
				iterator.remove();
			}
		}
	}

	private Map<Long, SparseVector> cleanMap(Map<Long, SparseVector> map) {
		Map<Long, SparseVector> newMap = new HashMap<Long, SparseVector>(map);
		Long[] array = {7l,47l,111l,154l,189l,190l,199l,211l,260l,273l,277l,362l,390l,394l,401l,426l,495l,556l,592l,594l,596l,599l,615l,616l,659l,668l,670l,674l,680l,716l,746l,750l,755l,775l,783l,796l,820l,823l,824l,841l,858l,897l,898l,899l,900l,901l,902l,903l,904l,905l,906l,907l,908l,909l,910l,911l,912l,913l,914l,915l,916l,918l,919l,920l,922l,923l,924l,925l,926l,928l,929l,931l,932l,933l,935l,936l,937l,938l,939l,940l,942l,943l,944l,945l,946l,947l,948l,949l,950l,951l,953l,954l,955l,956l,958l,959l,960l,961l,963l,964l,965l,966l,968l,969l,970l,971l,972l,973l,974l,976l,981l,982l,1007l,1008l,1009l,1010l,1011l,1012l,1014l,1017l,1019l,1021l,1022l,1023l,1024l,1025l,1026l,1028l,1029l,1030l,1031l,1032l,1035l,1040l,1067l,1068l,1069l,1070l,1073l,1076l,1077l,1078l,1080l,1082l,1083l,1084l,1085l,1086l,1103l,1104l,1125l,1136l,1152l,1154l,1156l,1157l,1161l,1162l,1164l,1178l,1193l,1194l,1201l,1203l,1204l,1206l,1207l,1208l,1209l,1212l,1214l,1219l,1221l,1226l,1230l,1232l,1234l,1235l,1237l,1244l,1247l,1248l,1250l,1251l,1252l,1253l,1254l,1256l,1260l,1262l,1263l,1267l,1269l,1272l,1276l,1277l,1278l,1281l,1282l,1283l,1284l,1287l,1292l,1301l,1303l,1304l,1329l,1331l,1333l,1337l,1339l,1340l,1341l,1345l,1348l,1350l,1367l,1371l,1380l,1381l,1386l,1387l,1388l,1419l,1421l,1423l,1496l,1502l,1520l,1534l,1572l,1628l,1774l,1811l,1815l,1849l,1913l,1922l,1924l,1926l,1927l,1929l,1930l,1931l,1932l,1933l,1934l,1935l,1936l,1937l,1938l,1939l,1940l,1943l,1944l,1945l,1947l,1948l,1949l,1950l,1951l,1952l,1953l,1954l,1955l,1963l,1964l,1997l,1998l,2009l,2010l,2013l,2015l,2016l,2017l,2018l,2019l,2025l,2031l,2032l,2034l,2035l,2037l,2038l,2040l,2043l,2047l,2049l,2051l,2055l,2056l,2066l,2067l,2074l,2078l,2080l,2085l,2090l,2095l,2096l,2098l,2099l,2102l,2109l,2131l,2132l,2138l,2160l,2163l,2176l,2177l,2178l,2179l,2180l,2181l,2182l,2183l,2184l,2185l,2186l,2187l,2200l,2201l,2202l,2203l,2204l,2207l,2208l,2209l,2210l,2211l,2212l,2214l,2215l,2216l,2218l,2220l,2221l,2222l,2223l,2229l,2238l,2239l,2258l,2285l,2299l,2303l,2308l,2351l,2361l,2362l,2363l,2365l,2389l,2398l,2409l,2425l,2488l,2489l,2495l,2498l,2511l,2520l,2521l,2522l,2523l,2524l,2526l,2527l,2528l,2529l,2530l,2531l,2532l,2533l,2536l,2537l,2554l,2559l,2565l,2584l,2602l,2612l,2632l,2635l,2636l,2637l,2640l,2644l,2646l,2647l,2648l,2649l,2650l,2651l,2652l,2653l,2654l,2656l,2657l,2660l,2661l,2663l,2664l,2665l,2666l,2667l,2669l,2670l,2726l,2727l,2728l,2729l,2730l,2731l,2732l,2746l,2747l,2779l,2780l,2781l,2782l,2783l,2784l,2785l,2788l,2789l,2814l,2819l,2821l,2823l,2847l,2853l,2854l,2856l,2857l,2862l,2863l,2866l,2870l,2871l,2874l,2877l,2895l,2896l,2899l,2901l,2904l,2905l,2920l,2921l,2922l,2923l,2925l,2927l,2932l,2933l,2935l,2936l,2937l,2939l,2940l,2941l,2944l,2946l,2947l,2948l,2949l,2951l,2954l,2967l,2969l,2971l,2974l,2981l,2983l,2984l,2991l,2993l,2994l,3011l,3012l,3015l,3022l,3024l,3025l,3026l,3027l,3028l,3030l,3032l,3035l,3037l,3038l,3040l,3049l,3058l,3061l,3062l,3066l,3069l,3073l,3074l,3075l,3076l,3086l,3089l,3092l,3093l,3095l,3096l,3097l,3099l,3119l,3121l,3122l,3132l,3133l,3134l,3135l,3136l,3139l,3140l,3143l,3144l,3151l,3152l,3153l,3154l,3167l,3168l,3171l,3172l,3178l,3194l,3195l,3196l,3198l,3199l,3200l,3201l,3202l,3204l,3205l,3207l,3209l,3215l,3216l,3217l,3220l,3224l,3230l,3231l,3232l,3244l,3245l,3283l,3284l,3292l,3294l,3296l,3305l,3306l,3307l,3309l,3311l,3314l,3315l,3330l,3332l,3333l,3334l,3336l,3337l,3339l,3340l,3341l,3343l,3344l,3345l,3348l,3349l,3350l,3351l,3359l,3362l,3363l,3364l,3365l,3367l,3368l,3369l,3371l,3372l,3373l,3375l,3376l,3377l,3380l,3384l,3396l,3405l,3406l,3414l,3415l,3417l,3419l,3420l,3421l,3427l,3430l,3431l,3435l,3445l,3447l,3451l,3458l,3462l,3467l,3468l,3469l,3470l,3471l,3472l,3473l,3475l,3485l,3486l,3490l,3492l,3494l,3498l,3503l,3504l,3506l,3507l,3508l,3516l,3519l,3520l,3522l,3533l,3542l,3545l,3546l,3548l,3549l,3551l,3559l,3560l,3583l,3584l,3585l,3588l,3589l,3590l,3599l,3600l,3601l,3602l,3603l,3604l,3605l,3606l,3607l,3610l,3612l,3622l,3627l,3628l,3629l,3630l,3632l,3633l,3634l,3635l,3638l,3639l,3641l,3642l,3643l,3644l,3645l,3648l,3651l,3653l,3654l,3655l,3656l,3658l,3659l,3670l,3671l,3674l,3675l,3676l,3678l,3681l,3682l,3702l,3724l,3729l,3730l,3733l,3735l,3736l,3737l,3738l,3739l,3741l,3742l,3744l,3758l,3759l,3760l,3769l,3771l,3772l,3775l,3776l,3779l,3780l,3781l,3782l,3788l,3789l,3792l,3801l,3803l,3804l,3806l,3807l,3808l,3812l,3813l,3814l,3818l,3832l,3833l,3836l,3845l,3847l,3849l,3850l,3866l,3870l,3871l,3872l,3873l,3875l,3878l,3899l,3922l,3923l,3924l,3926l,3927l,3928l,3929l,3930l,3931l,3932l,3933l,3934l,3957l,3958l,3963l,3964l,3965l,3966l,3984l,3985l,4044l,4045l,4046l,4049l,4051l,4064l,4065l,4184l,4185l,4186l,4187l,4188l,4189l,4190l,4192l,4194l,4195l,4196l,4208l,4212l,4218l,4263l,4274l,4278l,4282l,4283l,4287l,4292l,4294l,4296l,4298l,4316l,4319l,4320l,4323l,4324l,4325l,4327l,4328l,4329l,4330l,4331l,4337l,4338l,4339l,4340l,4349l,4356l,4357l,4359l,4360l,4363l,4394l,4395l,4399l,4401l,4402l,4403l,4405l,4406l,4411l,4412l,4413l,4414l,4416l,4419l,4420l,4422l,4423l,4424l,4426l,4427l,4428l,4431l,4432l,4433l,4434l,4436l,4437l,4438l,4440l,4441l,4444l,4463l,4476l,4479l,4643l,4687l,4689l,4695l,4704l,4705l,4708l,4709l,4710l,4712l,4754l,4756l,4767l,4768l,4785l,4786l,4789l,4790l,4791l,4794l,4795l,4796l,4797l,4798l,4799l,4800l,4801l,4802l,4803l,4804l,4805l,4806l,4810l,4811l,4813l,4829l,4853l,4854l,4855l,4857l,4863l,4905l,4907l,4908l,4910l,4911l,4912l,4914l,4916l,4917l,4918l,4920l,4922l,4923l,4924l,4925l,4927l,4928l,4930l,4932l,4942l,4943l,4944l,4945l,4947l,4948l,4953l,4966l,4969l,4970l,4972l,4982l,4984l,4998l,4999l,5000l,5002l,5003l,5005l,5007l,5017l,5019l,5021l,5022l,5035l,5036l,5037l,5056l,5057l,5058l,5060l,5063l,5072l,5084l,5085l,5087l,5088l,5090l,5091l,5094l,5097l,5098l,5099l,5100l,5105l,5113l,5114l,5115l,5116l,5117l,5119l,5121l,5122l,5123l,5124l,5126l,5132l,5140l,5141l,5142l,5143l,5144l,5145l,5147l,5148l,5149l,5155l,5156l,5165l,5167l,5168l,5169l,5177l,5209l,5227l,5228l,5230l,5231l,5232l,5233l,5234l,5238l,5247l,5262l,5263l,5275l,5289l,5290l,5291l,5292l,5301l,5302l,5304l,5328l,5333l,5336l,5341l,5352l,5353l,5354l,5355l,5356l,5368l,5369l,5371l,5372l,5373l,5374l,5375l,5376l,5381l,5382l,5383l,5384l,5385l,5386l,5392l,5393l,5394l,5395l,5396l,5397l,5398l,5399l,5405l,5406l,5408l,5410l,5411l,5412l,5429l,5431l,5434l,5436l,5440l,5468l,5469l,5472l,5473l,5474l,5487l,5488l,5489l,5490l,5491l,5492l,5493l,5494l,5495l,5496l,5497l,5498l,5499l,5511l,5519l,5522l,5544l,5545l,5549l,5550l,5551l,5552l,5554l,5560l,5583l,5586l,5590l,5593l,5595l,5599l,5600l,5601l,5603l,5604l,5638l,5639l,5640l,5641l,5642l,5644l,5649l,5651l,5659l,5660l,5661l,5692l,5693l,5695l,5708l,5722l,5733l,5735l,5745l,5754l,5778l,5795l,5799l,5800l,5801l,5802l,5825l,5826l,5828l,5830l,5834l,5836l,5864l,5867l,5868l,5881l,5891l,5892l,5899l,5922l,5963l,5965l,5966l,5972l,5974l,5975l,5977l,5979l,5981l,5983l,5984l,5986l,5990l,5997l,5998l,6019l,6020l,6021l,6023l,6029l,6030l,6031l,6032l,6035l,6047l,6048l,6051l,6052l,6054l,6064l,6065l,6066l,6072l,6073l,6139l,6170l,6172l,6176l,6178l,6181l,6182l,6183l,6184l,6208l,6225l,6226l,6228l,6229l,6230l,6231l,6232l,6233l,6234l,6236l,6237l,6245l,6247l,6254l,6256l,6257l,6258l,6271l,6273l,6276l,6277l,6301l,6302l,6305l,6307l,6316l,6317l,6331l,6354l,6355l,6356l,6357l,6358l,6386l,6390l,6391l,6392l,6394l,6395l,6396l,6397l,6398l,6401l,6403l,6404l,6407l,6408l,6409l,6410l,6411l,6412l,6413l,6414l,6416l,6421l,6422l,6426l,6428l,6429l,6430l,6431l,6432l,6433l,6434l,6435l,6437l,6438l,6441l,6443l,6446l,6447l,6449l,6450l,6451l,6452l,6453l,6455l,6456l,6457l,6458l,6459l,6460l,6462l,6465l,6467l,6469l,6470l,6477l,6478l,6480l,6495l,6497l,6498l,6499l,6500l,6509l,6511l,6512l,6513l,6514l,6515l,6520l,6521l,6522l,6524l,6527l,6528l,6530l,6532l,6533l,6561l,6562l,6573l,6576l,6578l,6579l,6581l,6584l,6585l,6599l,6604l,6605l,6607l,6609l,6610l,6611l,6613l,6630l,6639l,6643l,6645l,6646l,6649l,6650l,6652l,6654l,6655l,6656l,6657l,6660l,6663l,6665l,6666l,6669l,6684l,6716l,6724l,6725l,6727l,6728l,6730l,6732l,6738l,6739l,6741l,6743l,6746l,6748l,6777l,6779l,6781l,6783l,6784l,6785l,6787l,6798l,6808l,6813l,6821l,6825l,6826l,6830l,6836l,6848l,6849l,6852l,6854l,6856l,6858l,6859l,6900l,6903l,6907l,6910l,6911l,6912l,6913l,6918l,6919l,6920l,6921l,6923l,6924l,6925l,6926l,6960l,6963l,6967l,6970l,6980l,6981l,6982l,6984l,6985l,6986l,6987l,6988l,6990l,6995l,7001l,7008l,7013l,7024l,7029l,7031l,7043l,7049l,7050l,7051l,7052l,7053l,7055l,7056l,7058l,7059l,7060l,7061l,7062l,7063l,7064l,7065l,7067l,7068l,7069l,7070l,7071l,7072l,7073l,7075l,7076l,7077l,7078l,7079l,7080l,7081l,7082l,7084l,7085l,7086l,7088l,7089l,7091l,7092l,7093l,7095l,7104l,7106l,7107l,7111l,7115l,7116l,7119l,7121l,7122l,7124l,7126l,7128l,7130l,7131l,7132l,7135l,7136l,7178l,7179l,7180l,7181l,7182l,7183l,7184l,7188l,7194l,7195l,7200l,7204l,7205l,7206l,7207l,7209l,7210l,7211l,7212l,7213l,7214l,7215l,7216l,7217l,7218l,7219l,7221l,7222l,7224l,7225l,7227l,7230l,7231l,7234l,7236l,7237l,7238l,7239l,7241l,7243l,7244l,7245l,7247l,7249l,7252l,7253l,7272l,7273l,7274l,7275l,7279l,7280l,7281l,7288l,7289l,7290l,7300l,7301l,7302l,7303l,7304l,7308l,7309l,7311l,7312l,7327l,7328l,7329l,7330l,7331l,7333l,7334l,7335l,7336l,7341l,7344l,7357l,7370l,7386l,7388l,7389l,7391l,7392l,7394l,7396l,7397l,7398l,7402l,7405l,7406l,7407l,7414l,7416l,7420l,7474l,7479l,7482l,7484l,7485l,7491l,7493l,7523l,7560l,7564l,7565l,7569l,7571l,7577l,7580l,7581l,7582l,7583l,7585l,7586l,7587l,7614l,7617l,7619l,7638l,7650l,7697l,7698l,7700l,7702l,7703l,7705l,7706l,7720l,7723l,7748l,7751l,7756l,7757l,7761l,7762l,7764l,7766l,7771l,7772l,7774l,7792l,7802l,7808l,7813l,7814l,7817l,7820l,7821l,7822l,7826l,7828l,7831l,7832l,7833l,7834l,7835l,7836l,7838l,7840l,7881l,7882l,7883l,7885l,7886l,7888l,7889l,7890l,7891l,7892l,7893l,7894l,7895l,7896l,7897l,7898l,7899l,7900l,7901l,7913l,7914l,7916l,7917l,7919l,7920l,7921l,7922l,7923l,7924l,7925l,7926l,7933l,7935l,7936l,7937l,7938l,7939l,7940l,7941l,7942l,7944l,7945l,7946l,7948l,7949l,7953l,7954l,7958l,7979l,7980l,7984l,7989l,7990l,7992l,7993l,7994l,7995l,8003l,8004l,8008l,8009l,8015l,8017l,8033l,8039l,8042l,8044l,8056l,8057l,8069l,8094l,8125l,8126l,8136l,8137l,8138l,8139l,8140l,8143l,8149l,8153l,8154l,8167l,8187l,8188l,8189l,8190l,8191l,8194l,8195l,8196l,8197l,8198l,8199l,8202l,8203l,8206l,8207l,8222l,8225l,8227l,8228l,8232l,8235l,8236l,8238l,8239l,8252l,8253l,8256l,8257l,8259l,8261l,8262l,8263l,8264l,8267l,8272l,8290l,8291l,8292l,8295l,8302l,8330l,8331l,8334l,8336l,8337l,8338l,8340l,8381l,8384l,8385l,8388l,8391l,8395l,8399l,8401l,8403l,8404l,8405l,8407l,8410l,8422l,8423l,8424l,8427l,8446l,8447l,8450l,8451l,8452l,8453l,8456l,8459l,8460l,8461l,8462l,8463l,8465l,8477l,8480l,8481l,8482l,8483l,8484l,8486l,8487l,8488l,8491l,8492l,8494l,8496l,8499l,8502l,8503l,8507l,8511l,8512l,8516l,8518l,8519l,8522l,8524l,8525l,8540l,8542l,8543l,8544l,8571l,8572l,8583l,8584l,8586l,8592l,8595l,8596l,8600l,8601l,8606l,8608l,8609l,8611l,8612l,8613l,8615l,8616l,8617l,8618l,8620l,8626l,8635l,8647l,8649l,8650l,8651l,8652l,8654l,8657l,8660l,8661l,8664l,8669l,8670l,8672l,8673l,8675l,8677l,8678l,8679l,8681l,8682l,8683l,8684l,8685l,8686l,8687l,8688l,8690l,8692l,8693l,8694l,8695l,8696l,8700l,8711l,8712l,8714l,8715l,8716l,8718l,8719l,8724l,8725l,8726l,8727l,8728l,8734l,8735l,8738l,8739l,8740l,8744l,8745l,8748l,8751l,8752l,8754l,8756l,8757l,8758l,8759l,8761l,8762l,8763l,8765l,8766l,8767l,8769l,8771l,8772l,8773l,8774l,8775l,8778l,8780l,8781l,8785l,8786l,8787l,8788l,8789l,8794l,8796l,8818l,8821l,8822l,8838l,8840l,8844l,8848l,8852l,8856l,8875l,8876l,8879l,8881l,8882l,8886l,8889l,8890l,8894l,8896l,8897l,8899l,8900l,8920l,8921l,8923l,8924l,8926l,8928l,8932l,8986l,8988l,8989l,8990l,8991l,8992l,8993l,8997l,8998l,8999l,9001l,9003l,9007l,9008l,9011l,9012l,9014l,25737l,25744l,25746l,25750l,25753l,25759l,25763l,25764l,25769l,25771l,25773l,25774l,25777l,25783l,25795l,25797l,25802l,25805l,25826l,25827l,25828l,25832l,25833l,25839l,25840l,25841l,25855l,25856l,25866l,25868l,25870l,25878l,25885l,25886l,25888l,25891l,25893l,25898l,25899l,25900l,25903l,25904l,25905l,25906l,25908l,25914l,25916l,25924l,25929l,25930l,25931l,25934l,25937l,25938l,25942l,25943l,25945l,25947l,25951l,25952l,25954l,25960l,25961l,25962l,25963l,25964l,25971l,25975l,25977l,25995l,25996l,26002l,26005l,26007l,26012l,26048l,26049l,26052l,26055l,26059l,26064l,26067l,26073l,26078l,26079l,26082l,26084l,26085l,26094l,26096l,26101l,26110l,26111l,26112l,26116l,26119l,26122l,26123l,26124l,26128l,26131l,26133l,26134l,26136l,26138l,26139l,26142l,26144l,26147l,26148l,26150l,26151l,26152l,26155l,26158l,26159l,26163l,26170l,26171l,26176l,26178l,26181l,26185l,26187l,26189l,26199l,26203l,26208l,26211l,26222l,26225l,26226l,26228l,26230l,26231l,26240l,26242l,26246l,26249l,26251l,26258l,26265l,26269l,26270l,26271l,26283l,26285l,26289l,26294l,26301l,26302l,26306l,26313l,26318l,26323l,26326l,26336l,26339l,26345l,26349l,26350l,26359l,26360l,26364l,26366l,26371l,26375l,26379l,26386l,26388l,26393l,26394l,26398l,26404l,26413l,26422l,26431l,26491l,26734l,26819l,27109l,27528l,27790l,30701l,30712l,30721l,30783l,30942l,30949l,30952l,30954l,30994l,30996l,31030l,31042l,31079l,31086l,31104l,31107l,31109l,31116l,31156l,31160l,31188l,31193l,31247l,31255l,31260l,31270l,31309l,31347l,31349l,31467l,31522l,31528l,31588l,31590l,31613l,31617l,31689l,31737l,31742l,31854l,31930l,31950l,31973l,32049l,32060l,32076l,32139l,32141l,32149l,32151l,32156l,32158l,32160l,32162l,32166l,32174l,32179l,32211l,32261l,32325l,32329l,32345l,32347l,32349l,32361l,32369l,32371l,32375l,32381l,32383l,32387l,32395l,32525l,32617l,32625l,32632l,32649l,32674l,32677l,32721l,32781l,32792l,32844l,32853l,32862l,32875l,32882l,32892l,32954l,33036l,33072l,33113l,33126l,33191l,33193l,33237l,33312l,33451l,33564l,33608l,33781l,33905l,34002l,34018l,34065l,34170l,34177l,34321l,34359l,34583l,34608l,34643l,34645l,36056l,36553l,37211l,37287l,37335l,37545l,37976l,39369l,39429l,39474l,39659l,40833l,40870l,40973l,40988l,41014l,41136l,41336l,41627l,41699l,41831l,42094l,42217l,42556l,42681l,42783l,42900l,43011l,43710l,44073l,44168l,44587l,44657l,44911l,45335l,45611l,45662l,45679l,46258l,46653l,46664l,46803l,46855l,46901l,47274l,47306l,47330l,47493l,47615l,47721l,47723l,47728l,47810l,48061l,48301l,48374l,48649l,48899l,48909l,49007l,49200l,49355l,49571l,49688l,50229l,50253l,50354l,50356l,50477l,50619l,50970l,51014l,51277l,51638l,51857l,51911l,52101l,52352l,52767l,53737l,53887l,55020l,55159l,55895l,55926l,56015l,56067l,57476l,57480l,57484l,57550l,57690l,58059l,58439l,59157l,59173l,59832l,59834l,60201l,60309l,60411l,60484l,60487l,60745l,60990l,61279l,61434l,61470l,61634l,61934l,61937l,61967l,61970l,62000l,62153l,62198l,62245l,62526l,62669l,62801l,62803l,62916l,62920l,63121l,63187l,63239l,63647l,63676l,63760l,63768l,63772l,63793l,63989l,64245l,64273l,64283l,64338l,64497l,64903l,64906l,64926l,64930l,64959l,65091l};
		Set<Long> set = new HashSet<Long>(Arrays.asList(array));
		Iterator<Map.Entry<Long, SparseVector>> iterator = newMap.entrySet().iterator();
		while (iterator.hasNext()) {
			Map.Entry<Long, SparseVector> entry = iterator.next();
			if (set.contains(entry.getKey())) {
				iterator.remove();
			}
		}
		return newMap;
	}

	/*private Set<Long> getPopSet() {
		List<Long> list = popModel.getItemList();
		Set<Long> set = new HashSet<Long>(list.subList(0, Settings.POPULAR_ITEMS_SERENDIPITY_NUMBER));
		return set;
	}*/

	private List<Long> getMixedList(Map<Long, LinkedList<Container<Double>>> genresMap) {
		Set<Long> regSet = new HashSet<Long>();
		List<Long> list = new ArrayList<Long>();
		Set<Long> keys = genresMap.keySet();
		boolean full = true;
		while (full) {
			for (long key : keys) {
				Container<Double> container = null;
				if (genresMap.get(key).isEmpty()) {
					full = false;
					continue;
				}
				do {
					container = genresMap.get(key).poll();
				} while (!regSet.add(container.getId()) && !genresMap.get(key).isEmpty());
				list.add(container.getId());
				full = full || !genresMap.isEmpty();
			}
		}
		Collections.reverse(list);
		return list;
	}

	private LinkedList<Container<Double>> toContainerList(LinkedList<Long> list, SparseVector prediction) {
		LinkedList<Container<Double>> containerLinkedList = new LinkedList<Container<Double>>();
		for (long id : list) {
			if (prediction.containsKey(id)) {
				containerLinkedList.add(new Container<Double>(id, prediction.get(id)));
			}
		}
		Collections.sort(containerLinkedList);
		Collections.reverse(containerLinkedList);
		return containerLinkedList;
	}

	private Set<Long> getGenreSet(SparseVector userVector) {
		Set<Long> newFeatures = new HashSet<Long>();
		List<Container<Double>> list = new ArrayList<Container<Double>>();
		for (long key : userVector.keySet()) {
			list.add(new Container<Double>(key, userVector.get(key)));
		}
		Collections.sort(list);
		for (int i = 0; i < list.size(); i++) {
			if (i < Settings.GENRES_NUMBER || list.get(i).getValue() == 0.0) {
				newFeatures.add(list.get(i).getId());
			}
		}
		return newFeatures;
	}

	private LinkedList<Long> getItemsByGenre(long genre, Map<Long, SparseVector> map) {
		LinkedList<Long> list = new LinkedList<Long>();
		for (Map.Entry<Long, SparseVector> entry : map.entrySet()) {
			if (entry.getValue().containsKey(genre)) {
				list.add(entry.getKey());
			}
		}
		return list;
	}

	@Override
	public void score(long user, @Nonnull MutableSparseVector scores) {
		SparseVector userVector = getUserVector(user);
		List<Long> list = getRecommendations(user, userVector);
		for (VectorEntry e : scores.view(VectorEntry.State.EITHER)) {
			if (list.indexOf(e.getKey()) == -1) {
				scores.set(e, 0);
			} else {
				scores.set(e, list.indexOf(e.getKey()));
			}
		}
	}
}
