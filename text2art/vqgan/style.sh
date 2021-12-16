#!/bin/bash -eu

SRC_DIR=$(dirname $(realpath $0))
CODE_DIR=$(realpath $SRC_DIR/../..)

prompt=
exp_name=
styles="none artstation artstation_hd unreal deviant cryengine flickr behance painting oil watercolour pastel ink_wash acrylic hot_wax digital futuristic vintage sketch lines comic cartoon escher dali van_gogh picasso morris mona_lisa monet fresco printmaking abstract modern impressionism pop_art cubism surrealism contemporary fantasy graffiti anime art_deco art_nouveau bauhaus futureism expressionism minimalism geometric"

. ./scripts/parse_options.sh || exit 1;

[ -z "$prompt" ] && echo "please provide a prompt" && exit 1

exp_name=${exp_name:-$(date +"%Y%m%d")}

for style_name in $styles; do
    case $style_name in
        none) end_of_prompt="" ;;
        artstation) end_of_prompt=" #artstation" ;;
        artstation_hd) end_of_prompt=" trending on ArtStation HD" ;;
        unreal) end_of_prompt=" | Unreal Engine" ;;
        deviant) end_of_prompt=" | deviantart" ;;
        cryengine) end_of_prompt=" | CryEngine" ;;
        flickr) end_of_prompt=" | Flickr" ;;
        behance) end_of_prompt=" | Behance HD" ;;
        painting) end_of_prompt=" as a painting" ;;
        oil) end_of_prompt=" as an oil painting" ;;
        watercolour) end_of_prompt=" as a watercolour painting" ;;
        pastel) end_of_prompt=" as a pastel painting" ;;
        ink_wash) end_of_prompt=" as an ink wash painting" ;;
        acrylic) end_of_prompt=" as a vibrant acrylic painting" ;;
        hot_wax) end_of_prompt=" as a hot wax painting" ;;
        digital) end_of_prompt=" as a digital painting" ;;
        futuristic) end_of_prompt=" as a futuristic painting" ;;
        vintage) end_of_prompt=" #vintage #photo" ;;
        sketch) end_of_prompt=" as a sketch" ;;
        lines) end_of_prompt=" as black lines on white background" ;;
        comic) end_of_prompt=" in a comic #marvel" ;;
        cartoon) end_of_prompt=" as a cartoon" ;;
        escher) end_of_prompt=" in the style of M.C. Escher" ;;
        dali)  end_of_prompt=" in the style of Salvador Dali" ;;
        van_gogh)  end_of_prompt=" in the style of Van Gogh" ;;
        picasso)  end_of_prompt=" in the style of Pablo Picasso" ;;
        morris) end_of_prompt=" in the style of William Morris" ;;
        mona_lisa) end_of_prompt=" in the style of Mona Lisa" ;;
        monet) end_of_prompt=" in the style of the Claude Monet" ;;
        fresco) end_of_prompt=" in the style of 16th Century Fresco" ;;
        printmaking) end_of_prompt=" #printmaking" ;;
        abstract) end_of_prompt=" #abstract" ;;
        modern) end_of_prompt=" #modern" ;;
        impressionism) end_of_prompt=" #impressionism" ;;
        pop_art) end_of_prompt=" in the style of pop art" ;;
        cubism) end_of_prompt=" #cubism" ;;
        surrealism) end_of_prompt=" #surrealism" ;;
        contemporary) end_of_prompt=" #contemporary" ;;
        fantasy) end_of_prompt=" #fantasy" ;;
        graffiti) end_of_prompt=" #graffiti" ;;
        anime) end_of_prompt=" #anime" ;;
        art_deco) end_of_prompt=" in the style of art deco" ;;
        art_nouveau) end_of_prompt=" in the style of art nouveau" ;;
        bauhaus) end_of_prompt=" in the style of Bauhaus" ;;
        futureism)  end_of_prompt=" #futureism" ;;
        expressionism) end_of_prompt=" #expressionism" ;;
        minimalism) end_of_prompt=" #minimalism" ;;
        geometric) end_of_prompt=" #geometric" ;;
        *) end_of_prompt=" #$style_name" ;;        
    esac
    $CODE_DIR/text2art/vqgan/run.sh --WORK_DIR /exp/$(whoami)/text2art_styles/$exp_name/$style_name --prompt "$prompt$end_of_prompt" --image_name ${exp_name}_${style_name}
done

