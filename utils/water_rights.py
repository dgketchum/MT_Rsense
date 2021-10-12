from geopandas import read_file
from fuzzywuzzy import process, fuzz


def ap_fuzz(x, y):
    pext = process.extract(x, y, limit=1, scorer=fuzz.token_sort_ratio)[0]
    _str, score = pext[0], pext[1]
    return [_str, score]


def fuzzy_join(wr_shp, cad_shp, out_name):
    cad = read_file(cad_shp)
    cad = cad[~cad['OwnerName'].isnull()]
    wr_shp = read_file(wr_shp).drop(columns='geometry')
    cad['key'] = cad.OwnerName.apply(lambda x: ap_fuzz(x, wr_shp.ALLOWNERS))
    cad['fuzz_score'] = cad.key.apply(lambda x: x[1])
    cad['key'] = cad.key.apply(lambda x: x[0])
    df = cad.merge(wr_shp, left_on='key', right_on='ALLOWNERS')
    df.to_csv(out_name)
    df.to_file(out_name.replace('.csv', '.shp'))


def join_pou_pod():
    pass


if __name__ == '__main__':
    cadestral_fields = '/media/research/IrrigationGIS/Montana/water_rights/clark_fork_sid_32100_ownerparcel.shp'
    w_rights = '/media/research/IrrigationGIS/Montana/water_rights/irrRights_CFR_12334550.shp'
    out = '/media/research/IrrigationGIS/Montana/water_rights/cad_wr_merge_token_score.csv'
    fuzzy_join(w_rights, cadestral_fields, out)
# ========================= EOF ====================================================================
