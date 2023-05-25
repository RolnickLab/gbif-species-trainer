"""
Author         : Aditya Jain
Last modified  : May 23rd, 2023
About          : Calculates top1 accuracy for every taxon at different taxonomy levels
"""


def taxon_accuracy(data, label_info):
    """
    returns top1 accuracy for every taxon at different ranks
    """
    final_data = {}
    final_data[
        "About"
    ] = "[Top1 Accuracy, Total Test Points] for each taxon at different ranks"

    # Family
    family_list = label_info["family"]
    f_data = data["family"]
    family_data = {}

    for key in f_data.keys():
        family_data[family_list[key]] = [f_data[key][0], f_data[key][3]]

    family_data = dict(
        sorted(family_data.items(), key=lambda item: item[1], reverse=True)
    )
    final_data["family"] = family_data

    # Genus
    genus_list = label_info["genus"]
    g_data = data["genus"]
    genus_data = {}

    for key in g_data.keys():
        genus_data[genus_list[key]] = [g_data[key][0], g_data[key][3]]

    genus_data = dict(
        sorted(genus_data.items(), key=lambda item: item[1], reverse=True)
    )
    final_data["genus"] = genus_data

    # Species
    species_list = label_info["species"]
    s_data = data["species"]
    species_data = {}

    for key in s_data.keys():
        species_data[species_list[key]] = [s_data[key][0], s_data[key][3]]

    species_data = dict(
        sorted(species_data.items(), key=lambda item: item[1], reverse=True)
    )
    final_data["species"] = species_data

    return final_data
