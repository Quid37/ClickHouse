CREATE DATABASE IF NOT EXISTS test;
DROP TABLE IF EXISTS test.defaults;
CREATE TABLE IF NOT EXISTS test.defaults
(
    param1 Float64,
    param2 Float64
) ENGINE = Memory;
insert into test.defaults values (10.73744614,9.79931757),(9.70072286,10.63712519),(12.54639402,11.36933697),(9.28432111,14.25612833),(7.15810545,11.09798545),(6.00161925,11.57331404),(8.57770658,12.73230083),(11.25342075,9.44488283),(12.30830558,7.15693974),(11.75297284,9.15237157),(10.70000421,13.6225408),(13.80969478,8.50016754),(11.66776009,10.75167091),(7.22180622,10.86542074),(8.86066104,12.59098114),(9.27182315,5.36172349),(11.7588814,6.93907354),(9.78788726,10.04089),(9.24629271,11.49243616),(13.03287653,11.87629713),(8.17210245,7.61351783),(8.84450372,8.59285586),(11.48075113,13.51415649),(10.67131963,8.1285403),(12.23268287,15.88058065),(9.11763197,8.04966194),(10.40371567,8.85008344),(6.66709684,10.2832007),(9.84340923,10.00366727),(9.92785804,11.4347353),(10.9027221,8.57455898),(9.8659191,11.61294012),(13.17653657,9.42053601),(10.01473778,13.15732601),(9.89191132,9.77974463),(11.52558437,14.21076036),(11.13980036,8.88296733),(10.42291616,12.90819066),(12.07785478,11.59840703),(7.56218052,7.79332398),(13.0144777,12.44409116),(11.87768348,8.83921491),(7.62297701,11.90618886),(9.04639641,8.22869599),(10.73963124,11.18498912),(8.39057055,12.88110002),(8.10277322,7.99769165),(8.81966082,8.27489738),(10.66067449,10.39350841),(10.86622718,8.46734739),(7.47647153,8.50117337),(8.70808876,13.24215795),(11.63975677,10.00894659),(10.79227927,8.64692971),(6.77702805,11.7681239),(8.95816082,11.44097338),(10.29327757,8.02937473),(7.67872162,9.20639333),(10.06128684,8.17228671),(10.62882011,7.89877385),(11.1521952,8.88006692),(9.81222766,10.35002745),(7.59753884,8.6713338),(12.36361237,10.9444077),(10.67696025,9.60971544),(10.88041714,6.98419366),(9.31414009,10.98560116),(10.05222465,9.14558648),(10.66736361,7.18472225),(10.22123234,9.57636358),(11.5353634,10.92928204),(9.61886903,10.89838322),(11.10039186,7.80443237),(9.15671815,12.36431987),(10.07620618,7.692757),(10.17208684,12.15267038),(11.29803374,11.34419455),(6.92480595,9.70989012),(8.60958799,8.83375294),(9.19890067,12.35599927),(12.33780837,5.83625816),(14.85127655,15.35098341),(11.13325582,4.36670228),(12.35502376,6.41826696),(7.5265035,7.96842527),(10.80404348,7.69960333),(9.7698094,10.13987476),(10.67247465,8.30634224),(10.42695309,8.6818138),(11.22792852,8.52593682),(7.79504357,11.14594698),(9.200418,7.10516691),(11.90719505,9.15855129),(7.3257418,10.66242259),(9.29417184,10.35823449),(7.22312609,7.07803458),(7.11872214,7.89389879),(7.8222587,10.3527727),(9.29524571,11.55048464),(6.58952018,9.69043725),(-10.99382043,9.3000181),(-7.76988201,10.14536266),(-11.66042593,10.49429202),(-10.87189998,10.55818591),(-5.52787268,8.65942359),(-9.53783913,9.68632119),(-11.57170376,11.35210972),(-7.12100992,6.81173187),(-12.16469696,12.39135879),(-6.94365882,7.80214629),(-12.24739225,11.64194718),(-8.98042311,8.59966628),(-11.39317448,9.02606986),(-8.62553944,12.24194901),(-14.22680583,7.8748582),(-8.52074726,8.19675223),(-7.89925278,8.08604524),(-7.88351336,11.43212411),(-10.38692249,9.44966065),(-13.08407434,9.7402947),(-11.99934207,9.39929994),(-7.7723872,11.44156288),(-7.56902261,9.64490767),(-9.59217541,11.12531064),(-8.08659729,12.66203267),(-11.67408911,10.15111508),(-9.13946979,9.20750729),(-7.7566518,11.02544212),(-11.69311597,10.88034518),(-9.21993713,11.81906374),(-14.14503676,13.53580849),(-11.21386017,9.17705721),(-12.69034959,8.48830586),(-10.84265314,7.10898675),(-9.60610048,8.16838223),(-12.97271551,10.27862765),(-11.01387018,11.65458212),(-6.55207238,5.69629487),(-14.57735544,9.39659561),(-11.07226438,11.61103736),(-12.07361778,12.63027559),(-10.93686633,11.91928872),(-11.56263779,8.68431882),(-7.4365122,8.0577968),(-14.04555577,8.86156691),(-11.15579799,11.81809642),(-7.92379874,9.11113379),(-10.59083645,10.99442032),(-9.69249841,10.20881341),(-11.33807734,12.92411323),(-11.27782343,7.15729699),(-8.21544187,9.24511961),(-10.07924154,8.79123825),(-11.40708128,8.28690745),(-10.26466728,9.18866853),(-12.73758735,11.11285836),(-9.88701064,12.53064427),(-11.98303297,9.11198995),(-8.90971669,7.83990335),(-10.19841715,9.41337716),(-6.97081479,9.71609732),(-11.00761556,7.32075968),(-9.47955726,9.67922982),(-10.22432402,10.04549021),(-11.88312645,8.91708523),(-11.33772291,11.78141735),(-9.58605386,10.97853935),(-9.4262536,8.90495191),(-9.52343629,11.52107468),(-6.77427082,8.94814057),(-12.28876103,9.77555424),(-11.50014616,10.18035011),(-10.63396813,8.48269706),(-13.12792096,12.75449983),(-7.94945336,12.98109612),(-12.47382633,10.18591241),(-10.17443592,10.49299681),(-7.44930222,10.08854548),(-9.4637019,10.01289516),(-6.00259078,11.46642521),(-8.28186183,7.55846346),(-13.65875862,9.74655422),(-11.06234664,11.57460896),(-10.10525129,12.74471086),(-9.75276967,7.86575557),(-10.00934639,7.45768103),(-16.20521508,8.64086225),(-7.78652452,9.66022052),(-9.25648903,5.26958111),(-8.75796232,10.05968326),(-11.08677666,12.18605741),(-9.66088464,7.65764362),(-10.74827606,9.39358809),(-10.79994305,8.22121356),(-7.36597903,8.71148916),(-7.9175643,6.28810231),(-10.24338757,9.31392664),(-10.74767388,6.85870305),(-7.45728479,11.25389352),(-10.23985652,6.37836913),(11.22977961,-7.42106644),(9.34443194,-9.41079984),(9.5861104,-10.28674698),(8.92197142,-8.51496138),(8.59337887,-10.33440682),(9.41169224,-12.22858711),(8.75898571,-11.94575212),(13.0309728,-8.49504726),(6.48775004,-12.19095737),(12.30961006,-8.0331434),(7.76115052,-11.25643159),(11.04188944,-11.0487647),(13.17631362,-12.89871242),(12.12876878,-9.04761772),(9.09452259,-10.48013482),(9.07645932,-10.04085948),(8.93919741,-9.47460778),(8.07095142,-7.81446937),(10.20476561,-7.91642917),(12.45972277,-8.68729736),(8.14008627,-12.06550381),(6.00609678,-9.3477956),(10.46908567,-11.710469),(10.64242087,-11.8799362),(9.36262061,-14.37383346),(9.27584271,-13.36495262),(11.73109446,-8.7032307),(10.35273935,-9.80625948),(10.9763898,-10.22721835),(11.14756119,-6.66308747),(7.81786232,-10.73924483),(8.73298228,-10.48450138),(7.20312827,-9.26735143),(9.03343595,-11.12591262),(12.21534056,-9.27965983),(11.73314406,-13.23556129),(8.20430714,-11.17397262),(10.08636994,-7.62299361),(8.16390023,-9.06523015),(6.52779666,-9.53878733),(6.61136665,-8.73271462),(13.66913672,-10.55222763),(11.32667029,-9.74639624),(9.61889981,-10.50595581),(10.02097347,-13.42438514),(9.10363463,-8.00909898),(12.94261539,-9.72882626),(9.55901149,-8.19136129),(8.8908403,-12.2737754),(9.26104794,-8.26378002),(7.76218024,-11.27083756),(5.88661507,-7.5554306),(7.04503998,-9.65343045),(8.4972886,-8.22334637),(14.38334206,-10.04224558),(10.26168283,-8.3495938),(5.72619873,-11.61733121),(10.91534646,-7.9889419),(11.36636434,-6.43159572),(8.89694628,-7.69602406),(7.35479504,-12.53784915),(9.35966174,-11.84402202),(9.58362299,-9.3622663),(10.60590005,-13.18460972),(11.62214063,-11.63631647),(15.1064851,-11.32059962),(9.82168868,-10.47335345),(7.82646792,-9.23945867),(12.09475793,-9.73693996),(11.17368648,-6.1434999),(10.81987409,-9.6873509),(7.18292319,-7.77296827),(10.12548002,-10.85659936),(7.73586194,-12.26821037),(9.6966761,-8.16421418),(7.43140604,-8.5554451),(7.51032617,-12.68533856),(11.39540264,-9.80607848),(10.19794771,-12.53559019),(8.03855283,-8.76120048),(11.19542978,-10.49573131),(6.7132328,-10.06915249),(8.67865609,-10.09779899),(8.05719642,-13.37465013),(9.81925966,-9.05238566),(11.73792919,-11.59871436),(9.52835565,-8.02073039),(11.72715751,-11.66263285),(10.6366312,-9.91715089),(9.38001041,-8.75163914),(9.43728788,-10.85190452),(5.51636696,-9.3895359),(13.84482233,-10.37008117),(13.65435768,-10.17443251),(10.84485659,-7.87908907),(12.66808884,-9.38448604),(8.13881497,-8.19716273),(7.28710302,-11.73301995),(10.66104782,-7.53291996),(13.26565756,-5.63730528),(-9.41014585,-6.20946337),(-8.81842974,-7.96766148),(-13.15543235,-9.30778934),(-10.96281587,-8.50874654),(-7.03531757,-12.54863992),(-10.74128891,-9.51419197),(-14.74218067,-12.45041779),(-9.34253634,-10.11133036),(-8.92570771,-8.33837851),(-12.67288624,-8.85639542),(-8.95163129,-11.40061952),(-12.08314883,-6.74727395),(-6.5755237,-12.88582183),(-12.49862694,-7.82235105),(-10.21764725,-10.317968),(-11.70390644,-11.42751183),(-9.29220028,-10.24134775),(-12.57597222,-12.22842023),(-12.68471698,-8.47673456),(-10.98032079,-8.77825858),(-13.73589793,-11.53963782),(-8.12721159,-10.29593543),(-8.63053145,-9.0548728),(-9.63536863,-9.45751099),(-10.45182084,-11.27141409),(-8.05144267,-7.73278238),(-10.59009905,-9.64627052),(-11.67897016,-9.50659142),(-6.12250757,-11.09973379),(-13.45929578,-10.00096985),(-7.01197913,-8.06489791),(-8.36337601,-9.82037136),(-9.33082635,-11.709283),(-12.02811714,-9.25179879),(-6.47328525,-10.35347496),(-7.10348548,-8.43746916),(-9.69078011,-10.19160959),(-7.71236983,-9.03303048),(-4.71181787,-7.66527716),(-11.55154104,-8.63325342),(-6.37034214,-9.40715566),(-10.54409596,-5.74461277),(-8.00499679,-7.9550515),(-10.66183276,-9.71851021),(-9.86476558,-11.57400354),(-10.78097115,-14.03032572),(-10.82308474,-12.04135935),(-9.7869734,-10.34657614),(-11.5243565,-8.37531569),(-11.72295014,-11.42506931),(-9.41480739,-10.12789306),(-10.41465617,-12.22761567),(-8.74442847,-6.09129749),(-9.73086168,-9.47354076),(-10.49279489,-5.11692866),(-10.44030799,-5.58798205),(-9.50394467,-8.71359039),(-8.25906477,-11.66060013),(-10.85110477,-6.55917239),(-11.4708653,-11.13100922),(-12.60275604,-6.2839826),(-5.80180494,-11.19895927),(-8.28722702,-10.38247456),(-8.8112688,-14.63893352),(-13.12962672,-12.20905326),(-9.60600935,-11.21094635),(-8.55706195,-8.25943946),(-11.58572303,-11.11213138),(-6.97044177,-9.46027223),(-12.49144147,-8.53469609),(-8.80276675,-10.77250273),(-12.06016003,-10.71050816),(-14.44088229,-9.96182931),(-12.17096716,-8.18755215),(-12.50957084,-10.1690166),(-11.30669211,-10.62690621),(-11.85037891,-8.27288736),(-10.2551235,-9.79847378),(-10.29669796,-10.73107426),(-9.91494363,-10.31972486),(-10.98445861,-5.41278205),(-12.94123479,-8.49258725),(-10.01599818,-8.8721005),(-12.12970392,-7.91457435),(-11.69756899,-7.3402348),(-10.52974611,-13.07343463),(-6.42668956,-9.27905078),(-12.62339854,-9.0017173),(-10.54906613,-5.42317101),(-12.47509905,-10.94121985),(-9.24314326,-10.16151478),(-10.903536,-12.11086351),(-10.86850358,-5.98212386),(-6.70742725,-12.73913415),(-11.17288575,-9.03805836),(-11.44604621,-6.05741942),(-9.56292607,-10.25456795),(-9.36390218,-7.68903409),(-8.06001768,-10.52943106),(-10.11214745,-9.57904679);

DROP TABLE IF EXISTS test.model;
CREATE TABLE test.model ENGINE = Memory AS SELECT IncrementalClusteringState(4)(param1, param2) AS state FROM test.defaults;

DROP TABLE IF EXISTS test.answer;
create table test.answer engine = Memory as
select rowNumberInAllBlocks() as row_number, ans from
(with (select state from test.model) as model select evalMLMethod(model, param1, param2) as ans from test.defaults);

select row_number from test.answer where ans = (select ans from test.answer where row_number = 0) order by row_number;
select row_number from test.answer where ans = (select ans from test.answer where row_number = 100) order by row_number;
select row_number from test.answer where ans = (select ans from test.answer where row_number = 200) order by row_number;
select row_number from test.answer where ans = (select ans from test.answer where row_number = 300) order by row_number;
