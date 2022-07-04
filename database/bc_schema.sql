CREATE TABLE IF NOT EXISTS `bc` 
(
    `user_id` INT NOT NULL AUTO_INCREMENT,
    `diagnosis` CHAR NOT NULL,
    `radius_mean` float NOT NULL,
    `texture_mean` float NOT NULL,
    `perimeter_mean` float NOT NULL,
    `area_mean` float NOT NULL,
    `concavity_mean` float NOT NULL,
    `concave points_mean` float NOT NULL,
    `area_se` float NOT NULL,
    `radius_worst` float NOT NULL,
    `texture_worst` float NOT NULL,
    `perimeter_worst` float NOT NULL,
    `area_worst` float NOT NULL,
    `smoothness_worst` float NOT NULL,
    `compactness_worst` float NOT NULL,
    `concavity_worst` float NOT NULL,
    `concave points_worst` float NOT NULL,
    `symmetry_worst` float NOT NULL,
    PRIMARY KEY (`user_id`)
)
ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE utf8mb4_unicode_ci;