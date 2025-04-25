using Base.Threads
using DelimitedFiles
using CSV, DataFrames, BenchmarkTools

# Ruta base a los datasets
ruta_base = "../data/CMU_Artic"
ruta_destino = "../data/phonemas"

acento_por_dataset = Dict(
    "cmu_us_bdl_arctic" => "en-us",     # USA Male
    "cmu_us_slt_arctic" => "en-us",     # USA Female
    "cmu_us_clb_arctic" => "en-us",     # USA Female
    "cmu_us_rms_arctic" => "en-us",     # USA Male
    "cmu_us_jmk_arctic" => "en-us",     # Canadian male, there is no hay en-ca
    "cmu_us_awb_arctic" => "en-sc",     # Scottish male
    #"cmu_us_ksp_arctic" => "en"        # indio, usar genérico
)

function phonemize_word(word, voice)
    cmd = `espeak -q --ipa -v $voice "$word"`
    try
        ipa = String(read(cmd))
        return strip(ipa)
    catch
        return ""
    end
end

# Detectar si la palabra contiene un fonema deseado
function detectar_fonema(ipa)
    fonemas = ["æ", "ɪ", "ʌ", ]
    for f in fonemas
        if occursin(f, ipa)
     #       println("Fonema: $f para IPA: $ipa")  
            return f
        end
    end

    return nothing
end


# Procesar una línea del archivo txt
function procesar_linea(linea, voice, dataset)
    #println("Procesando línea: $linea")  # Imprime la línea
    m = match(r"\(\s*(\S+)\s+\"(.*?)\"\s*\)", linea)
    if m === nothing
        return []
    end
    audio_id, oracion = m.captures
    palabras = split(lowercase(oracion), r"[^a-z]+")
    resultado = []

    for palabra in palabras
        #println("Palabra: $palabra")  # Imprime cada palabra
        ipa = phonemize_word(palabra, voice)
        fonema = detectar_fonema(ipa)
        # Solo se añadirá si pertenece a la lista de fonemas
        if fonema !== nothing
            push!(resultado, (palabra, fonema, audio_id, dataset))
        end
    end

    return resultado
end

# Procesar todos los datasets
# MAIN: Procesar todos los datasets
function procesar_datasets(ruta_base::String)
    datasets = collect(pairs(acento_por_dataset))
    Threads.@threads for i in 1:length(datasets)
        dataset, voice = datasets[i]

        ruta_txt = joinpath(ruta_base, dataset, "etc", "txt.done.data")
        isfile(ruta_txt) || continue  # Saltar si no existe

        println("📂 Procesando $dataset con voz $voice...")
        lineas = readlines(ruta_txt)
        resultados_locales = Vector{Tuple{String, String, String, String}}()

        lock_global = Threads.SpinLock()

        Threads.@threads for j in 1:length(lineas)
            resultado = procesar_linea(lineas[j], voice, dataset)
            lock(lock_global) do
                append!(resultados_locales, resultado)
            end
        end

        println("✅ Se extrajeron $(length(resultados_locales)) palabras para $dataset.")

        # Guardar CSV por dataset
        df = DataFrame(resultados_locales, [:Palabra, :Fonema, :AudioID, :Dataset])
        archivo_salida = "phonemas_$(dataset).csv"
        CSV.write(archivo_salida, df)
        println("💾 Guardado: $archivo_salida")
    end
end


println(Threads.nthreads())
#MAIN
tiempo = @elapsed procesar_datasets(ruta_base)
println("⏱️ Tiempo total: $tiempo segundos")
