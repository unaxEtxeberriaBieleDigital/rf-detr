from rfdetr.variants import RFDETRLarge


def main():
    model = RFDETRLarge(pretrain_weights=r"C:\Users\u.etxeberria\Downloads\lat1_large.pth")
    pred = model.predict(
        r"C:\Users\u.etxeberria\Downloads\lat_dataset\training_dataset\test\5704_115661_Left_native_tile_0_704.png",
        include_source_image=False,
        return_query_embeddings=True,
    )
    # pred.query_embeddings: [num_detections, num_decoder_layers * hidden_dim]
    print(pred)
    print(pred.query_embeddings.shape)


if __name__ == "__main__":
    main()
