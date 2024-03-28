import pandas as pd
import matplotlib.pyplot as plt

dic = {
-1: [('kubrick', 0.0004526786901279503), ('turing', 0.00044066048661203485), ('joker', 0.00044018969798385806), ('kane', 0.00042212439297049177), ('citizen', 0.0004186930586689115), ('welles', 0.00041799635338163647), ('jaws', 0.00041041176013147503), ('shark', 0.0004075034943431588), ('ballet', 0.0004032810728041472), ('lester', 0.0003990097952164387), ('frodo', 0.00039510312302581816), ('mendes', 0.00038837873396314936)], 
0: [('bligh', 0.0061104672347980895), ('mutiny', 0.005228698022083954), ('northup', 0.005109351026306515), ('wallace', 0.004551903519570438), ('dunkirk', 0.00434277717362291), ('ruggles', 0.004310768166792146), ('robin', 0.004268577128997672), ('braveheart', 0.004209005330184459), ('bounty', 0.003873669996168911), ('queeg', 0.003830064400093719), ('slave', 0.0036248624353303824), ('flynn', 0.003480903862383435)], 
1: [('margo', 0.007457680311388955), ('eve', 0.0072700902041661205), ('evelyn', 0.006214877600137358), ('hawking', 0.005636398527563869), ('stark', 0.005556447057471753), ('gaz', 0.004744280613710567), ('woodward', 0.004523214647316522), ('monty', 0.004309019885435169), ('nic', 0.004006311773976134), ('yeoh', 0.003904989371895178), ('bagel', 0.003879085246757832), ('waymond', 0.003879085246757832)], 
2: [('gehrig', 0.007038846703543528), ('darlington', 0.004877879749239368), ('prewitt', 0.004736999128420673), ('maggio', 0.00444679916674111), ('farnsworth', 0.004325086160731919), ('todd', 0.004282683499101273), ('lan', 0.003988320445475146), ('fogg', 0.003988320445475146), ('corkle', 0.003665237985841704), ('cantinflas', 0.003665237985841704), ('yankees', 0.0036325774573723463), ('woody', 0.003577034113408766)], 
3: [('patton', 0.00314088864356656), ('chaplin', 0.002999439982651886), ('casablanca', 0.002747217930319723), ('doss', 0.0025416825502187186), ('platoon', 0.0025073473224920505), ('kovic', 0.0022837642736280423), ('rick', 0.00212141666451553), ('saving', 0.0021201088723323412), ('alamo', 0.002009611292127402), ('malick', 0.001880474776764596), ('soldiers', 0.0018787518862066921), ('barber', 0.001849682451018726)], 
4: [('maximus', 0.0032539806841057352), ('cleopatra', 0.003068417396182694), ('gladiator', 0.002745961493287754), ('shakespeare', 0.0027399508704394537), ('hamlet', 0.0027309991565314636), ('commodus', 0.0026998315704498174), ('romeo', 0.002417991071089044), ('rome', 0.0023615528370629513), ('moulin', 0.0022674070731517823), ('zorba', 0.002234387659585483), ('antony', 0.002217334370258701), ('juliet', 0.0021601698088468896)], 
5: [('juror', 0.010141056042573687), ('gantry', 0.009630659082629667), ('spade', 0.00889591418264245), ('falconer', 0.006899659793080598), ('arrowsmith', 0.006597918033515919), ('fran', 0.005608729525331536), ('jurors', 0.005603836207641701), ('lulu', 0.005378696757835513), ('skippy', 0.0053259230745656555), ('dingle', 0.0050139587121004695), ('senate', 0.004815329541314833), ('falcon', 0.004668590945189678)], 
6: [('ziegfeld', 0.0036220117072485267), ('astaire', 0.0035679119653304354), ('bernadette', 0.003003593387026418), ('tevye', 0.0028511416221630764), ('poppins', 0.0028243443723913815), ('cohan', 0.0028007199277975147), ('trapp', 0.0026072835609055986), ('capra', 0.002425874452626938), ('malley', 0.0024248566251579796), ('salzburg', 0.0023467987916257703), ('cagney', 0.0023211166458001597), ('deeds', 0.00228185575517947)], 
7: [('friedkin', 0.002162711598798952), ('exorcist', 0.0019755089807406276), ('django', 0.0018718550101801278), ('forrest', 0.0018567741082356588), ('mac', 0.0018295784714049493), ('pulp', 0.0018100468153570187), ('blatty', 0.0016772293108360619), ('regan', 0.001633347717789498), ('lumet', 0.0015833847623609664), ('duvall', 0.0015082279182249527), ('milk', 0.001488813385772368), ('gump', 0.00148847235428932)], 
8: [('panther', 0.001910071988836738), ('jojo', 0.0017208767211726388), ('godfather', 0.001673733768353982), ('corleone', 0.0016343920144507413), ('cia', 0.001540076006830271), ('ark', 0.0015240758712100713), ('wakanda', 0.0014010825085198847), ('schindler', 0.0013943494720938525), ('waititi', 0.0013887583576305332), ('challa', 0.001363623762595737), ('inception', 0.001347016992476565), ('raiders', 0.0013068075300145781)], 
9: [('juno', 0.0046411777318068545), ('sammy', 0.003456723921632494), ('tár', 0.0028992769608555337), ('deaf', 0.0026833721347139985), ('parasite', 0.0026802966263080024), ('ki', 0.0026713280438177625), ('cassie', 0.0026604782931760225), ('lydia', 0.0025983778361496593), ('bong', 0.0025579106742892087), ('pádraic', 0.0025402355601748827), ('babe', 0.0024180364285430315), ('jonze', 0.0022766182464289547)], 
10: [('rocky', 0.006335572412055563), ('micky', 0.004086782502305796), ('stallone', 0.0039221304140126595), ('champ', 0.0036441730800784076), ('graffiti', 0.0036319913008440855), ('furiosa', 0.0034548096106880206), ('raging', 0.003431433381510274), ('lamotta', 0.0033840208210140573), ('bull', 0.0032024748387913357), ('dink', 0.0031726521813545947), ('villa', 0.0029541152775385323), ('fury', 0.002937559850838275)], 
11: [('beast', 0.003376014023803304), ('dalton', 0.0032016520533601944), ('booth', 0.0029183200132398236), ('pi', 0.0027659629025109057), ('hughes', 0.0026920141355492077), ('manson', 0.0026407257241848176), ('chazelle', 0.002578048049146575), ('tate', 0.0024825274510246824), ('avatar', 0.00247004556078829), ('toy', 0.0024336618331258397), ('pixar', 0.0022987374438406144), ('belle', 0.0022805604129317375)], 
12: [('higgins', 0.0051836516218762025), ('eliza', 0.004525410146082399), ('mildred', 0.004317012825065036), ('johnnie', 0.003966556353053105), ('mame', 0.003910185809049409), ('sikes', 0.0035791603135239044), ('streisand', 0.003539784602445572), ('veda', 0.00332462134894287), ('oliver', 0.003166972585121449), ('belinda', 0.003095952421223431), ('lina', 0.0030540603866206243), ('hepburn', 0.0029747396819060105)], 
13: [('jo', 0.0037492386030284087), ('erin', 0.0035549696949782657), ('skeeter', 0.0035240284518676143), ('jacy', 0.003431220432191113), ('stokowski', 0.003431220432191113), ('bogdanovich', 0.0033245449958139374), ('hellman', 0.003181192276579536), ('brockovich', 0.0030795806401098997), ('miniver', 0.0029992172663137677), ('regina', 0.0029818702285033657), ('julia', 0.002936309543110098), ('minny', 0.0028349697523096396)]
}

for key, value in dic.items():
    #print(key, value)
    #print("\n")
    valores = []
    palavras = []
    for element in value:
        palavras.append(element[0])
        valores.append(element[1])
    #print(palavras, valores)
    #print("\n")
    plt.figure(figsize=(12,8))
    plt.barh(palavras, valores, color='green')
    plt.xlim(0, 0.01)
    plt.xlabel("Palavras")
    plt.ylabel("Porcentagem")
    plt.title("Topic {}".format(key))
    plt.savefig("IMGS BERTOPIC/inglês/topic {}".format(key))
    valores.clear()
    palavras.clear()