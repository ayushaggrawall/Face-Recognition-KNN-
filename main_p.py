# path = '/home/*/Desktop/non_news_rss.json'

#----------Import this module if you want to train KMeans clustering----------

# import training
# training.main(path)

# #-----------------------------------------------------------------------------


#----------Get Euc. dist. between cluster centroids and also get minimum dist. among two centroids of clusters----------

import euc_dist
euc_dist.main()

#-----------------------------------------------------------------------------


#----------To verify if cluster points are in centroid----------

# import verify
# verify.main(path)

#-----------------------------------------------------------------------------



#----------Pass a query face (128 bit) to find nearest cluster and check nearest datapoint----------

import query
import time
start=time.time()

x=[{"$numberDouble":"-0.35000473260879517"},{"$numberDouble":"-0.6810246706008911"},{"$numberDouble":"-0.6299816370010376"},{"$numberDouble":"-0.6837102770805359"},{"$numberDouble":"-1.3555655479431152"},{"$numberDouble":"1.9415647983551025"},{"$numberDouble":"-0.45182129740715027"},{"$numberDouble":"-0.21082967519760132"},{"$numberDouble":"1.0619258880615234"},{"$numberDouble":"-1.3959150314331055"},{"$numberDouble":"-0.4919472634792328"},{"$numberDouble":"-1.3383320569992065"},{"$numberDouble":"0.6231734156608582"},{"$numberDouble":"0.03856942057609558"},{"$numberDouble":"0.19033989310264587"},{"$numberDouble":"-0.22623813152313232"},{"$numberDouble":"1.1592116355895996"},{"$numberDouble":"0.09652024507522583"},{"$numberDouble":"0.1503312587738037"},{"$numberDouble":"-1.0588819980621338"},{"$numberDouble":"1.0969619750976562"},{"$numberDouble":"0.05944898724555969"},{"$numberDouble":"0.2390754222869873"},{"$numberDouble":"-0.7594559192657471"},{"$numberDouble":"-1.0693594217300415"},{"$numberDouble":"-0.9764097332954407"},{"$numberDouble":"-0.27827778458595276"},{"$numberDouble":"-0.8486369848251343"},{"$numberDouble":"-0.9037821888923645"},{"$numberDouble":"1.241233229637146"},{"$numberDouble":"0.12163926661014557"},{"$numberDouble":"0.1982276886701584"},{"$numberDouble":"-0.514701247215271"},{"$numberDouble":"1.535872220993042"},{"$numberDouble":"0.7740784883499146"},{"$numberDouble":"-0.1819053441286087"},{"$numberDouble":"0.09980684518814087"},{"$numberDouble":"-0.44028961658477783"},{"$numberDouble":"1.7838492393493652"},{"$numberDouble":"-0.2661137580871582"},{"$numberDouble":"0.9161348342895508"},{"$numberDouble":"-0.04183094948530197"},{"$numberDouble":"-0.4962655007839203"},{"$numberDouble":"-1.8574320077896118"},{"$numberDouble":"1.1484980583190918"},{"$numberDouble":"0.1677781343460083"},{"$numberDouble":"1.6383466720581055"},{"$numberDouble":"-1.152684211730957"},{"$numberDouble":"1.0221329927444458"},{"$numberDouble":"0.011139456182718277"},{"$numberDouble":"-0.5814396739006042"},{"$numberDouble":"0.500311017036438"},{"$numberDouble":"-1.3376989364624023"},{"$numberDouble":"-0.8923106789588928"},{"$numberDouble":"0.7220635414123535"},{"$numberDouble":"-0.07725660502910614"},{"$numberDouble":"-0.288423627614975"},{"$numberDouble":"-0.9274035692214966"},{"$numberDouble":"0.31408366560935974"},{"$numberDouble":"-0.22640040516853333"},{"$numberDouble":"1.2521775960922241"},{"$numberDouble":"-0.32863014936447144"},{"$numberDouble":"1.432982087135315"},{"$numberDouble":"0.43317341804504395"},{"$numberDouble":"0.8096439838409424"},{"$numberDouble":"-0.03830048441886902"},{"$numberDouble":"0.417866051197052"},{"$numberDouble":"0.8794601559638977"},{"$numberDouble":"-1.8184713125228882"},{"$numberDouble":"-1.145670771598816"},{"$numberDouble":"0.1276630461215973"},{"$numberDouble":"0.7742561101913452"},{"$numberDouble":"0.9815875291824341"},{"$numberDouble":"-2.8098528385162354"},{"$numberDouble":"2.117567300796509"},{"$numberDouble":"0.4709574580192566"},{"$numberDouble":"-1.6155424118041992"},{"$numberDouble":"-0.2221435159444809"},{"$numberDouble":"-0.6186299920082092"},{"$numberDouble":"-0.5088316798210144"},{"$numberDouble":"-1.163489580154419"},{"$numberDouble":"1.3761602640151978"},{"$numberDouble":"0.3756771385669708"},{"$numberDouble":"0.2476940155029297"},{"$numberDouble":"-0.7894148826599121"},{"$numberDouble":"0.9833148717880249"},{"$numberDouble":"-1.182601809501648"},{"$numberDouble":"0.36354994773864746"},{"$numberDouble":"-1.4019403457641602"},{"$numberDouble":"-0.7606787085533142"},{"$numberDouble":"1.4585883617401123"},{"$numberDouble":"1.0300397872924805"},{"$numberDouble":"-0.5715906620025635"},{"$numberDouble":"0.3250398635864258"},{"$numberDouble":"0.9446196556091309"},{"$numberDouble":"1.1126421689987183"},{"$numberDouble":"-0.44305527210235596"},{"$numberDouble":"1.107975959777832"},{"$numberDouble":"-0.032659076154232025"},{"$numberDouble":"1.4531526565551758"},{"$numberDouble":"1.4054807424545288"},{"$numberDouble":"0.6023568511009216"},{"$numberDouble":"-0.6714813709259033"},{"$numberDouble":"0.7212168574333191"},{"$numberDouble":"-0.3528284430503845"},{"$numberDouble":"0.45098787546157837"},{"$numberDouble":"-1.5802146196365356"},{"$numberDouble":"1.72651207447052"},{"$numberDouble":"-1.0570707321166992"},{"$numberDouble":"-1.184557557106018"},{"$numberDouble":"-0.8688845634460449"},{"$numberDouble":"-1.0732585191726685"},{"$numberDouble":"0.048299819231033325"},{"$numberDouble":"-0.8247315287590027"},{"$numberDouble":"-0.7942439317703247"},{"$numberDouble":"-0.895240306854248"},{"$numberDouble":"0.11495685577392578"},{"$numberDouble":"0.2053644061088562"},{"$numberDouble":"-0.8933081030845642"},{"$numberDouble":"-0.07082071900367737"},{"$numberDouble":"-0.6869916915893555"},{"$numberDouble":"0.7410644292831421"},{"$numberDouble":"2.1064789295196533"},{"$numberDouble":"-1.4959757328033447"},{"$numberDouble":"-0.9279530048370361"},{"$numberDouble":"-0.8800253868103027"},{"$numberDouble":"-1.0426535606384277"},{"$numberDouble":"-1.1729164123535156"}]
dd=[]
for i in x:
    dd.append(float(i['$numberDouble']))
query.main(dd)
end=time.time()
print(end-start)

#-----------------------------------------------------------------------------
