# StyleTTS2 Complete Architecture Diagram

This document contains comprehensive Mermaid diagrams showing how the entire StyleTTS2 repository works, based on the actual code implementation.

## Overview: High-Level Architecture

```mermaid
graph TB
    subgraph "Input"
        Text[Text Input]
        RefAudio[Reference Audio<br/>Optional]
    end

    subgraph "Auxiliary Models<br/>(Pre-trained, Frozen)"
        TA[Text Aligner<br/>Creates Text-Mel Alignment]
        PE[Pitch Extractor<br/>Extracts F0]
    end

    subgraph "Core Model Components"
        TE[Text Encoder<br/>Encodes phonemes]
        BERT[PL-BERT<br/>Phoneme embeddings]
        SE[Style Encoder<br/>Acoustic style]
        PE2[Predictor Encoder<br/>Prosodic style]
        PP[Prosody Predictor<br/>Duration + F0 + Energy]
        DIFF[Diffusion Model<br/>Style generation]
        DEC[Decoder<br/>Waveform generation]
    end

    subgraph "Output"
        Audio[Generated Speech<br/>24kHz]
    end

    Text --> TA
    Text --> TE
    Text --> BERT
    RefAudio --> SE
    RefAudio --> PE2
    
    TA -->|Alignment| PP
    TE -->|Text Features| PP
    BERT --> DIFF
    BERT --> PP
    SE --> DIFF
    PE2 --> DIFF
    PE2 --> PP
    
    PP -->|Duration, F0, N| DEC
    TE -->|Aligned Features| DEC
    SE -->|Acoustic Style| DEC
    DIFF -->|Predicted Style| DEC
    
    DEC --> Audio

    style TA fill:#e1f5ff
    style PE fill:#e1f5ff
    style SE fill:#fff4e1
    style PE2 fill:#fff4e1
    style DIFF fill:#ffe1f5
    style DEC fill:#e1ffe1
```

## Component Interaction Diagram

```mermaid
graph LR
    subgraph "Auxiliary Models"
        TA[Text Aligner<br/>ASRCNN+ASRS2S]
        PE[Pitch Extractor<br/>JDCNet]
    end

    subgraph "Text Processing"
        TE[Text Encoder<br/>CNN]
        BERT[PL-BERT]
        BE[BERT Encoder<br/>Linear]
    end

    subgraph "Style Extraction"
        SE[Style Encoder<br/>Acoustic]
        PE2[Predictor Encoder<br/>Prosodic]
    end

    subgraph "Prosody Modeling"
        DE[Duration Encoder]
        DP[Duration Predictor]
        F0N[F0 & N Predictor]
    end

    subgraph "Generation"
        DIFF[Diffusion<br/>Style Gen]
        DEC[Decoder<br/>Vocoder]
    end

    TA -->|Alignment| DE
    TE -->|Features| DEC
    BERT --> BE
    BE --> DE
    PE2 --> DE
    PE2 --> F0N
    DE --> DP
    DE --> F0N
    SE --> DIFF
    PE2 --> DIFF
    BERT --> DIFF
    DIFF -->|Style| DEC
    DP -->|Duration| DEC
    F0N -->|F0, N| DEC

    style TA fill:#e1f5ff
    style PE fill:#e1f5ff
    style SE fill:#fff4e1
    style PE2 fill:#fff4e1
    style DIFF fill:#ffe1f5
    style DEC fill:#e1ffe1
```

## Training Stage 1 Flow (train_first.py)

```mermaid
graph LR
    subgraph "Input"
        T[Text Tokens]
        M[Mel Spectrogram]
        W[Waveform]
    end

    subgraph "Text Alignment"
        TA[Text Aligner<br/>ASRCNN]
        TA -->|s2s_attn| Attn[Attention Matrix]
        Attn -->|maximum_path| MonoAttn[Monotonic Alignment]
    end

    subgraph "Feature Extraction"
        TE[Text Encoder] -->|t_en| ASR[ASR Features<br/>t_en @ alignment]
        M -->|Extract F0| PE[Pitch Extractor] -->|F0_real| F0[F0]
        M -->|Extract Style| SE[Style Encoder] -->|s| Style[Style Vector]
        M -->|log_norm| N[N_real Energy]
    end

    subgraph "Generation"
        ASR --> D[Decoder]
        F0 --> D
        N --> D
        Style --> D
        D -->|y_rec| Out[Generated Waveform]
    end

    subgraph "Discriminators"
        Out --> MPD[MPD]
        Out --> MSD[MSD]
        W --> MPD
        W --> MSD
    end

    T --> TA
    T --> TE
    M --> TA
    M --> SE
    M --> PE

    style TA fill:#e1f5ff
    style PE fill:#e1f5ff
    style SE fill:#fff4e1
    style D fill:#e1ffe1
```

## Training Stage 2 Flow (train_second.py)

```mermaid
graph TB
    subgraph "Input"
        T[Text Tokens]
        M[Mel Spectrogram]
        W[Waveform]
    end

    subgraph "Text Alignment"
        TA[Text Aligner] -->|s2s_attn_mono| Align[Alignment Matrix]
    end

    subgraph "Text & BERT Processing"
        T --> TE[Text Encoder] -->|t_en| ASR[ASR Features]
        T --> BERT[PL-BERT] -->|bert_dur| BE[BERT Encoder] -->|d_en| DE[Duration Encoder]
    end

    subgraph "Style Extraction"
        M --> SE[Style Encoder] -->|s| AcousticStyle[Acoustic Style]
        M --> PE[Predictor Encoder] -->|s_dur| ProsodicStyle[Prosodic Style]
    end

    subgraph "Prosody Prediction"
        DE -->|Duration| Dur[Predicted Duration]
        DE -->|p_en| F0N[F0Ntrain]
        ProsodicStyle --> F0N
        F0N -->|F0_fake, N_fake| F0NOut[F0 & N Curves]
    end

    subgraph "Ground Truth Extraction"
        M -->|Extract| PE2[Pitch Extractor] -->|F0_real| F0Real[F0_real]
        M -->|log_norm| NReal[N_real]
    end

    subgraph "Generation"
        ASR --> D[Decoder]
        F0NOut --> D
        AcousticStyle --> D
        D -->|y_rec| Out[Generated Waveform]
    end

    subgraph "Diffusion Model"
        BERT --> DIFF[Diffusion]
        AcousticStyle -->|Concatenate| STrg[s_trg]
        ProsodicStyle --> STrg
        STrg --> DIFF
        DIFF -->|Denoised| SPred[s_pred]
    end

    subgraph "Losses"
        Out -->|vs| W
        F0NOut -->|vs| F0Real
        F0NOut -->|vs| NReal
        Dur -->|vs| Align
        Out --> MPD[MPD]
        Out --> MSD[MSD]
        W --> MPD
        W --> MSD
    end

    T --> TA
    M --> TA
    Align --> ASR
    Align --> DE

    style TA fill:#e1f5ff
    style PE2 fill:#e1f5ff
    style SE fill:#fff4e1
    style PE fill:#fff4e1
    style DIFF fill:#ffe1f5
    style D fill:#e1ffe1
```

## Inference Flow (Based on Demo Notebooks)

```mermaid
graph TB
    subgraph "Input"
        Text[Input Text]
        RefAudio[Reference Audio<br/>Optional]
    end

    subgraph "Text Processing"
        Text --> Phon[Phonemizer] --> Tokens[Text Tokens]
        Tokens --> TE[Text Encoder] -->|t_en| TEnc[Text Features]
        Tokens --> BERT[PL-BERT] -->|bert_dur| BE[BERT Encoder] -->|d_en| DE[Duration Encoder]
    end

    subgraph "Style Generation"
        RefAudio -->|If provided| RefMel[Reference Mel]
        RefMel --> SE[Style Encoder] -->|ref_s| RefStyle[Reference Style]
        RefMel --> PE[Predictor Encoder] -->|ref_s_dur| RefProsodic[Reference Prosodic]
        
        BERT --> DIFF[Diffusion Sampler<br/>num_steps=5]
        RefStyle --> DIFF
        DIFF -->|s_pred| PredStyle[Predicted Style<br/>256-dim]
        PredStyle -->|Split| AcousticStyle[s_acoustic<br/>128-dim]
        PredStyle -->|Split| ProsodicStyle[s_prosodic<br/>128-dim]
        
        AcousticStyle -->|Blend with ref| FinalAcoustic[Final Acoustic Style]
        ProsodicStyle -->|Blend with ref| FinalProsodic[Final Prosodic Style]
    end

    subgraph "Duration & Prosody Prediction"
        DE -->|With s_prosodic| DurPred[Duration Prediction]
        DurPred -->|Create Alignment| Align[Predicted Alignment Matrix]
        TEnc -->|Align| ASR[Aligned ASR Features]
        
        DE -->|p_en| F0N[F0Ntrain]
        FinalProsodic --> F0N
        F0N -->|F0_pred, N_pred| F0NOut[F0 & N Curves]
    end

    subgraph "Waveform Generation"
        ASR --> Decoder[Decoder]
        F0NOut --> Decoder
        FinalAcoustic --> Decoder
        Decoder -->|Generated Audio| Output[Output Waveform<br/>24kHz]
    end

    style DIFF fill:#ffe1f5
    style SE fill:#fff4e1
    style PE fill:#fff4e1
    style Decoder fill:#e1ffe1
```

## Detailed Data Flow with Tensor Shapes

```mermaid
graph TB
    subgraph "Input Processing"
        Text["Text Tokens<br/>[B, T_text]"]
        Mel["Mel Spectrogram<br/>[B, 80, T_mel]"]
        Wave["Waveform<br/>[B, T_wav]"]
    end

    subgraph "Text Alignment"
        TA["Text Aligner<br/>ASRCNN + ASRS2S"]
        Mel --> TA
        Text --> TA
        TA -->|"s2s_attn<br/>[B, T_text, T_mel]"| Attn["Attention Matrix"]
        Attn -->|"maximum_path"| MonoAttn["Monotonic Alignment<br/>[B, T_text, T_mel]"]
    end

    subgraph "Text Encoding"
        TE["Text Encoder<br/>CNN"]
        Text --> TE
        TE -->|"t_en<br/>[B, 512, T_text]"| TEnc["Text Features"]
        TEnc -->|"@ alignment"| ASR["Aligned ASR<br/>[B, 512, T_mel//2]"]
    end

    subgraph "BERT Processing"
        BERT["PL-BERT"]
        Text --> BERT
        BERT -->|"bert_dur<br/>[B, T_text, 768]"| BE["BERT Encoder<br/>Linear"]
        BE -->|"d_en<br/>[B, 512, T_text]"| DE["Duration Encoder"]
    end

    subgraph "Style Extraction"
        SE["Style Encoder"]
        PE2["Predictor Encoder"]
        Mel --> SE
        Mel --> PE2
        SE -->|"s<br/>[B, 128]"| AcousticStyle["Acoustic Style"]
        PE2 -->|"s_dur<br/>[B, 128]"| ProsodicStyle["Prosodic Style"]
    end

    subgraph "Prosody Prediction"
        MonoAttn --> DE
        ProsodicStyle --> DE
        DE -->|"Duration<br/>[B, T_text]"| Dur["Predicted Duration"]
        DE -->|"p_en<br/>[B, 512, T_mel//2]"| F0N["F0Ntrain"]
        ProsodicStyle --> F0N
        F0N -->|"F0_fake<br/>[B, T_mel]"| F0Out["F0 Curve"]
        F0N -->|"N_fake<br/>[B, T_mel]"| NOut["N Curve"]
    end

    subgraph "Ground Truth Extraction"
        PE["Pitch Extractor"]
        Mel --> PE
        PE -->|"F0_real<br/>[B, T_mel]"| F0Real["F0 Ground Truth"]
        Mel -->|"log_norm"| NReal["N Ground Truth<br/>[B, T_mel]"]
    end

    subgraph "Waveform Generation"
        ASR --> DEC["Decoder<br/>HiFiGAN/iSTFTNet"]
        F0Out --> DEC
        NOut --> DEC
        AcousticStyle --> DEC
        DEC -->|"y_rec<br/>[B, T_wav]"| Output["Generated Audio"]
    end

    subgraph "Diffusion Model"
        BERT --> DIFF["Diffusion Sampler"]
        AcousticStyle -->|"Concatenate"| STrg["s_trg<br/>[B, 256]"]
        ProsodicStyle --> STrg
        STrg --> DIFF
        DIFF -->|"s_pred<br/>[B, 256]"| PredStyle["Predicted Style"]
    end

    style TA fill:#e1f5ff
    style PE fill:#e1f5ff
    style SE fill:#fff4e1
    style PE2 fill:#fff4e1
    style DIFF fill:#ffe1f5
    style DEC fill:#e1ffe1
```

## Component Details

### Text Aligner (Pre-trained)
- **Location**: `Utils/ASR/models.py`
- **Architecture**: ASRCNN (CNN encoder) + ASRS2S (Attention-based Seq2Seq)
- **Input**: Mel spectrogram + Text tokens
- **Output**: Attention alignment matrix `[batch, text_length, mel_length]`
- **Purpose**: Maps text tokens to mel spectrogram frames
- **Usage**: Creates alignment for duration prediction and text-to-mel alignment

### Pitch Extractor (Pre-trained)
- **Location**: `Utils/JDC/model.py`
- **Architecture**: JDCNet (Joint Detection and Classification Network)
- **Input**: Mel spectrogram `[batch, 1, n_mels, time]`
- **Output**: F0 values per frame `[batch, time]`
- **Purpose**: Extracts fundamental frequency for pitch control
- **Usage**: Provides ground truth F0 during training

### Style Encoders
- **Location**: `models.py:139` (StyleEncoder class)
- **Architecture**: CNN with 4 ResBlk layers + Global Average Pooling
- **Two Types**:
  1. **Style Encoder**: Extracts acoustic style (timbre, voice quality)
  2. **Predictor Encoder**: Extracts prosodic style (F0 patterns, rhythm, duration)
- **Input**: Mel spectrogram `[batch, 1, n_mels, time]`
- **Output**: Style vector `[batch, style_dim]` (typically 128-dim)

### Prosody Predictor
- **Location**: `models.py:440`
- **Components**:
  - **DurationEncoder**: Style-conditioned LSTM for duration prediction
  - **Duration Projection**: Predicts duration per text token
  - **F0Ntrain**: Predicts F0 and energy (N) curves
- **Input**: Text tokens, prosodic style, alignment
- **Output**: Duration, F0 curve, N (energy) curve

### Decoder
- **Location**: `Modules/hifigan.py` or `Modules/istftnet.py`
- **Architecture**: HiFiGAN or iSTFTNet vocoder
- **Input**: 
  - Aligned text features (ASR)
  - F0 curve
  - Energy (N) curve
  - Acoustic style vector
- **Output**: Generated waveform (24kHz)

### Diffusion Model
- **Location**: `Modules/diffusion/`
- **Purpose**: Generates style vectors from text embeddings
- **Input**: BERT embeddings, optional reference style
- **Output**: Predicted style vector (256-dim: 128 acoustic + 128 prosodic)
- **Usage**: Enables style transfer and diverse prosody generation

## Data Flow Summary

1. **Text → Phonemes**: Text is converted to phonemes using phonemizer
2. **Phonemes → Tokens**: Phonemes are tokenized into integer IDs
3. **Text Alignment**: Text aligner creates alignment between text and mel frames
4. **Text Encoding**: Text encoder processes tokens into features
5. **Style Extraction**: Style encoders extract acoustic and prosodic styles
6. **Prosody Prediction**: Prosody predictor generates duration, F0, and energy
7. **Alignment**: Text features are aligned to mel frames using predicted duration
8. **Waveform Generation**: Decoder generates audio from aligned features + F0 + N + style
9. **Discrimination**: Discriminators (MPD, MSD, WavLM) provide adversarial training signals

